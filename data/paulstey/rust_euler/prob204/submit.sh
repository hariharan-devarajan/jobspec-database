#!/bin/bash

#
# In order to submit this job to Slurm, run the command below from the terminal.
#   sbatch submit.sh

#SBATCH --time=1:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --constraint=48core
#SBATCH --cpus-per-task=48
#SBATCH --mem=24G
#SBATCH --job-name prob205 
#SBATCH --output prob204-%j.out
#SBATCH --error prob204-%j.err

#-------- End of Slurm commands -------------------------

module load rust 

cargo run --release
