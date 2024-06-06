#!/bin/bash
 
#SBATCH --nodes=1
#SBATCH --partition=gpuhgx
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=3

#SBATCH --mail-type=ALL
#SBATCH --mail-user=n_jaku01@uni-muenster.de

module load palma/2021b
module load Singularity
module load CUDA/11.6.0


singularity run --nv --bind .:/code open3d.sif train_essen.py "$1"
