#!/bin/bash

#SBATCH --account=researchers
#SBATCH --job-name=lmeval_bloom7B    # Job name
#SBATCH --output=../logs/R-%x.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --error=../logs/R-%x.%j.err     # Error handling
#SBATCH --nodes=1                # Total number of nodes requested
#SBATCH --cpus-per-task=8        # Schedule 8 cores (includes hyperthreading)
#SBATCH --mem=48G
#SBATCH --constraint="gpu_rtx8000|gpu_a100_40gb|gpu_v100" # Use either a v100 or a100
#SBATCH --gres=gpu:1      #v100:1 or a100_40gb:1 on brown
#SBATCH --time=0-02:00:00          # Run time (hh:mm:ss) 
#SBATCH --partition=red,brown    # Run on either the red or brown queue

#srun hostname

# module load Anaconda3
source activate lmeval # Not working???

nvidia-smi


model_path="bigscience/bloom-7b1"
model_name="bloom-7b1"


echo $model_name

lm_eval --model hf --model_args "pretrained=$model_path" --tasks "know_dist" --batch_size auto --max_batch_size 64 --device cuda:0 --output_path "../results/$model_name" --num_fewshot 0

