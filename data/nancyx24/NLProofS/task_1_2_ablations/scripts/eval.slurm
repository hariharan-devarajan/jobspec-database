#!/bin/bash
#SBATCH --job-name=nosearch   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:a5000:1            # number of gpus per node
#SBATCH --mem=16G
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=abiramg@princeton.edu
#SBATCH --output=../eval/nosearch.out

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/entailment_bank
conda activate entbank
python eval/run_scorer.py --task "task_2" --split test --prediction_file ../NLProofS/prover/lightning_logs/nosearch/results_test.tsv --output_dir ../ablation1/eval/nosearch/ --bleurt_checkpoint ../bleurt-large-512/
