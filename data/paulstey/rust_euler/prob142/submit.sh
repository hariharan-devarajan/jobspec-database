#!/bin/bash

#
# In order to submit this job to Slurm, run the command below from the terminal.
# `sbatch submit.sh`

#SBATCH --time=8:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --constraint=cascade
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name prob142 
#SBATCH --output prob142-%j.out
#SBATCH --error prob142-%j.err

#-------- End of Slurm commands -------------------------

module load rust 

cargo run --release
