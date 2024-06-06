#!/bin/bash

#SBATCH --job-name=nas        ## Name of the job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=23:59:59
#SBATCH --partition=gpu ##GPU run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=nas5.out    ## Output file 

source ~/.bashrc
conda deactivate
conda activate new

## Load the python interpreter
module load python

## Execute the python script
srun python nas.py
