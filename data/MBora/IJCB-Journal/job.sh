#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem 32000M
#SBATCH -t 0-11:59 # time (D-HH:MM)
#SBATCH --job-name="lstm"
#SBATCH -o ../Slurm/out%j.txt
#SBATCH -e ../Slurm/err%j.txt
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
nvidia-smi
conda env list
source activate cave
spack load cuda/gypzm3r
spack load cudnn
srun python3 ./run.py
