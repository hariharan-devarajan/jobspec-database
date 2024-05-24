#!/bin/bash

#SBATCH --partition=amd
#SBATCH --job-name=sample_ind
#SBATCH --account=pi-dachxiu
#SBATCH --output=../log/ind_output.txt
#SBATCH --error=../log/ind_error.txt
#SBATCH --time=1:00:00
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4

module load python

python3 ./src/empirical/indstock_new.py
