#!/bin/bash

#
# In order to submit this job to Slurm, run the command below from the terminal.
#   sbatch submit.sh

#SBATCH --time=0:10:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=80G
#SBATCH --constraint=48core
#SBATCH --job-name prob124 
#SBATCH --output prob124-%j.out
#SBATCH --error prob124-%j.err

#-------- End of Slurm commands -------------------------

module load rust 

cargo run --release 
