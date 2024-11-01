import os
import json
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--experiment_id", type=int, default="0",
                            help="id of json file")
parser.add_argument("--run", type=str, default="idle",
                            help="run to perform on experiment <id>")
parser.add_argument("--gpu_id", type=str, default="",
                            help="gpu IDs")
parser.add_argument("--config", type=str, default="",
                            help="config instruction")

args = parser.parse_args()
print(args)

print("Experiment ID: " + str(args.experiment_id))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

results_dir = '/om2/user/xboix/robustness/'
full_results_dir = results_dir + 'convex_robustness/'

if not args.run == 'config':

    with open(full_results_dir + 'configs/' + str(args.experiment_id) + '.json') as config_file:
        config = json.load(config_file)

    config['model_dir'] = full_results_dir + config['model_name']
    config['results_dir'] = results_dir

    with open(results_dir + 'configs_datasets/' + str(config["data_set_id"]) + '.json') as config_file:
        config_dataset = json.load(config_file)

    if args.run == 'train':
        import runs.train as run
        run.train(config)
    elif args.run == 'test':
        import runs.test as run
        run.test(config)

else:
    import runs.config_experiments_vision as run_exp

    if args.config == 'generate':
        run_exp.config_experiments(full_results_dir)
    elif args.config == 'generate_datasets':
        import runs.config_datasets as run
        run.config_datasets(results_dir)
    elif args.config == 'check':
        experiment_list = run_exp.config_experiments(full_results_dir, create_json=False)
        run_exp.check_uncompleted(full_results_dir, experiment_list)