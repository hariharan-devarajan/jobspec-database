import os, pickle, argparse, re
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from data import AMPDataset
from train import evalpred, getperf, perf_multi_label
from model import BERTAMP



def load_model_presets(rslt_dir: str):
    with open(os.path.join(rslt_dir, "arguments.txt"), "r+") as f:
        while f.readline() != 'model_presets:\n': pass
        mdl_presets = eval(f.readline())
    return mdl_presets


def load_task_labels(rslt_dir:str):
    with open(os.path.join(rslt_dir, "arguments.txt"), "r+") as f:
        for line in f:
            if 'labels: ' in line:
                match_obj = re.match(r'labels: (.*)', line, re.M|re.I)
                task_lab = match_obj.group(1)
                task_lab = eval(task_lab) if task_lab != 'AMP' else task_lab
                break
    return task_lab


def extract_embedding(dataloader, model, use_cuda=True):
    """function for extracting embedding.

    Executing the embedding of self-supervised model.

    Args:
        dataloader: pytorch DataLoader for handling evaluation.
        model: nn.Module to evaluate.
        use_cuda: whether use cuda.

    Returns:
        Evaluation results including average loss and average ACC.

        total_loss / cnt, total_acc / cnt
    """
    model.eval()
    all_embed = []
    desc = "Extracting embeddings"
    for i, batch_items in tqdm(enumerate(dataloader), total=dataloader.__len__(), leave=False, desc=desc):
        input_ids = batch_items['input_ids']
        input_mask = batch_items['input_mask']
        trg = batch_items['targets']
        if use_cuda:
            input_ids, input_mask, trg = input_ids.cuda(), input_mask.cuda(), trg.cuda()
        _, embed = model.forward(input_ids, input_mask)
        all_embed.append(embed.cpu().detach())
    return torch.cat(all_embed)


if __name__ == "__main__":
    ## Setting parser
    parser = argparse.ArgumentParser(description="Evaluating AMP classification")
    parser.add_argument('--path', required=True, type=str,
                    help='The result path for evaluation')
    parser.add_argument('--cuda', default=True, type=str,
                        help='Whether use cuda, defualt: True')
    
    args = parser.parse_args()
    ## model's presets and state to load (generated by train.py)
    RSLT_DIR = args.path
    USE_CUDA = args.cuda

    model_presets = load_model_presets(args.path)
    task_label = load_task_labels(args.path)

    print("Task labels:", str(task_label))

    if task_label == 'AMP':
        model = BERTAMP(**model_presets)
    else:
        model = BERTAMP(**model_presets, num_labels=len(task_label))
    if USE_CUDA:
        model = model.cuda()
        device_ids = 0
    print(os.path.join(RSLT_DIR, "final_models_evals.pkl"))
    model_dict = torch.load(os.path.join(RSLT_DIR, "final_models_evals.pkl"))
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    model.load_state_dict(model_dict['model_state_dict'])

    ## Load test data
    print("Loading data for evaluation...", flush=True, end=" ")
    if task_label == 'AMP':
        testdata = pd.read_csv("data/test/{:s}".format("stage-1.csv" if task_label == "AMP" else "mtl.csv"))
    else:
        testdata = pd.read_csv("data/test/mtl.csv")
    testset = AMPDataset(testdata, task_label=task_label)
    testloader = DataLoader(testset,
                            batch_size=128,
                            collate_fn=testset.collate_fn)
    print("Complete!")

    ## Get prediction results from test data
    print("Evaluation Process...", flush=True)
    testprob, testpred, testtrue = evalpred(testloader, model, multi_label=False if task_label == 'AMP' else True)
    if task_label == "AMP":
        test_performance = getperf(testprob, testpred, testtrue)
    else:
        print(testprob.shape)
        print(testtrue.shape)
        mtl_tab, mtl_cms, mtl_rocs = perf_multi_label(testprob, testtrue, label_names=task_label, thresholds=0.5)
        test_performance = {
            'mtl_perf': mtl_tab,
            'mtl_confusion_matrix': mtl_cms,
            'mtl_rocs': mtl_rocs
        }
    with open(os.path.join(RSLT_DIR, "test_result.pkl"), "wb") as f:
        pickle.dump(test_performance, f, pickle.HIGHEST_PROTOCOL)
    print("Complete!", flush=True)
    ## Save the embedding output
    print("Generating embedded outputs...", flush=True)
    embeddings = extract_embedding(testloader, model, use_cuda=True).numpy()
    embeddings = pd.DataFrame(embeddings, 
                              columns=["node{:d}".format(k) for k in range(embeddings.shape[1])])
    if task_label == "AMP":
        embeddings = pd.concat([testdata["Id"], embeddings, testdata["Label"]], axis=1)
    else:
        embeddings = pd.concat([testdata["Id"], embeddings, testdata.loc[:, task_label]], axis=1)
    embeddings.to_csv(os.path.join(RSLT_DIR, "test_embeddings.csv"), index=False)
    print("Complete!", flush=True)