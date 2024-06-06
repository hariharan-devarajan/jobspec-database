#!/bin/bash 
#SBATCH -o slurm-%j-%N.out
#SBATCH -A hpc0006177081
#SBATCH --partition=C032M0256G
#SBATCH -J txtbak
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --qos=low

module load anaconda/2-4.4.0.1
python  txtbak.py
