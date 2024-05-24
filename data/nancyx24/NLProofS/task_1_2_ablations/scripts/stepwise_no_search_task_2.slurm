#!/bin/bash
#SBATCH --job-name=stepwise_no_search_task_2      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:a5000:1            # number of gpus per node
#SBATCH --mem=16G
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all        # send email whenever job does anything
#SBATCH --mail-user=nancyx@princeton.edu
#SBATCH --output=stepwise_no_search_task_2.out

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/NLProofS
conda activate nlproofs
cd prover
python main.py test --config cli_task2_stepwise_t5-large.yaml --ckpt_path ../../weights/task2_stepwise.ckpt --model.proof_search false
