#!/bin/bash
#SBATCH --job-name=extract_data_from_hf    # Job name
#SBATCH -o logs/minestral-350m-en-it-07012024/extract_data_from_hf-job.out              # Name of stdout output file
#SBATCH -e logs/minestral-350m-en-it-07012024/extract_data_from_hf-job.err              # Name of stderr error file
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=4       # number of threads per task
#SBATCH --time 00:50:00          # format: HH:MM:SS

#SBATCH -A IscrB_medit

module load profile/deeplrn culturax/2309

# export OMP_PROC_BIND=true
export HF_DATASETS_CACHE=$WORK/hf_cache

source ~/llmfoundry-cuda-flash-attn2-env/bin/activate


# ~/llmfoundry-cuda-flash-attn2-env/bin/python scripts/data_prep/extract_hf_to_jsonl.py \
#     --dataset_path /leonardo/prod/data/ai/culturax/2309/en \
#     --path_to_save /leonardo_work/IscrB_medit/culturax/extracted/350M-model/en-v2/ \
#     --max_samples 10_000_000 --streaming --split_size 500_000 


~/llmfoundry-cuda-flash-attn2-env/bin/python /leonardo/home/userexternal/rorland1/llm-foundry/scripts/data_prep/extract_hf_to_jsonl.py \
    --dataset_path /leonardo/prod/data/ai/culturax/2309/it \
    --path_to_save /leonardo_work/IscrB_medit/culturax/extracted/350M-model/it/ \
    --max_samples 10_000_000 --streaming --split_size 500_000
