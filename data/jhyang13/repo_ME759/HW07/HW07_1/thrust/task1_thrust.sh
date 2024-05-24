#!/bin/bash

#SBATCH --job-name=task1_thrust
#SBATCH --partition=instruction
#SBATCH --time=00-00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=task1_thrust.out

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.8.0 
module load gcc/11.3.0

nvcc task1_thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1_thrust

./task1_thrust 10

#./task1_thrust {n}



