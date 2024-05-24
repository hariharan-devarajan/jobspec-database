#!/bin/bash
#SBATCH --job-name=1_0
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:a5000:1            # number of gpus per node
#SBATCH --mem=16G
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=all          # send email when job ends
#SBATCH --mail-user=abiramg@princeton.edu
#SBATCH --output=../outputs/1_0.out

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/NLProofS
conda activate nlproofs
cd prover
python main.py test --config cli_task2_stepwise_t5-large.yaml --log_name "1_0" --model.num_beam_groups 1 --model.diversity_penalty 0 --ckpt_path ../../weights/task2_stepwise.ckpt --model.verifier_weight 0.5 --model.verifier_ckpt ../../weights/task2_verifier.ckpt --model.proof_search true 