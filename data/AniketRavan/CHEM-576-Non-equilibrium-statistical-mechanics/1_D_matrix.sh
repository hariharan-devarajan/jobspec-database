#!/bin/bash
#SBATCH  -p normal
#SBATCH --mem=50g
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -J langevin_overdamped
#SBATCH -o langevin_overdamped
echo "Loading module"
#module load PyTorch/0.4.0-IGB-gcc-4.9.4-Python-3.6.1
echo "Loaded module. Running python"
module load Python/3.6.1-IGB-gcc-4.9.4
python 1_D_matrix.py
