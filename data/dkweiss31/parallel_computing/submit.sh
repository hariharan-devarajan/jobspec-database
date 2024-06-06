#!/bin/bash

#SBATCH --partition=day
#SBATCH --job-name=parallel
#SBATCH -o output-%a.txt -e errors-%a.txt
#SBATCH --array=0-5
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=daniel.weiss@yale.edu

NUM_LIST=($(seq 0 1 5))

echo "rng seed = " ${NUM_LIST[${SLURM_ARRAY_TASK_ID}]}

module load miniconda

conda activate qram_fidelity
python expensive_job.py --N=1000 --r=${NUM_LIST[${SLURM_ARRAY_TASK_ID}]}
