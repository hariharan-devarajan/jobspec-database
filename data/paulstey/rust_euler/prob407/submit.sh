#!/bin/bash

#
# In order to submit this job to Slurm, run the command below from the terminal.
#   sbatch submit.sh

#SBATCH --time=12:30:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=80G
#SBATCH --constraint=48core
#SBATCH --job-name prob407 
#SBATCH --output prob407-%j.out
#SBATCH --error prob407-%j.err

#-------- End of Slurm commands -------------------------

module load rust 

cargo run 
