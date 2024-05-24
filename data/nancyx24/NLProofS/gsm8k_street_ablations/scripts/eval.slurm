#!/bin/bash
#SBATCH --job-name=eval_scone   # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:a5000:1      # number of gpus per node
#SBATCH --mem=16G
#SBATCH --time=08:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all         # send email when job ends
#SBATCH --mail-user=abiramg@princeton.edu
#SBATCH --output=../eval/scone_prover_ent_verifier.out

module purge
conda init bash
source ~/.bashrc
cd /n/fs/nlp-abiramg/entailment_bank
conda activate entbank
python eval/run_scorer_scone_gsm8.py --task "scone" --split test --prediction_file "../NLProofS/prover/lightning_logs/ent_prover_ent_verifier_scone_test/results_test.tsv" --output_dir ../ablation3/eval/ent_prover_ent_verifier_scone/ --bleurt_checkpoint ../bleurt-large-512/