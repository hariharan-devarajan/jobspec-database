#!/bin/bash
#SBATCH --job-name=AsmQC
#SBATCH --account=nn9305k
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ajkarloss@gmail.com 

## Array Job 
#SBATCH --array=1-6

## Set up job environment:
source /cluster/bin/jobsetup

module load Miniconda3/4.4.10

source activate MultiQC
time python MultiQC.py
source deactivate MultiQC