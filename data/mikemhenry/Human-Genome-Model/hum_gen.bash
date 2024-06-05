#!/bin/bash
#SBATCH --mail-user=mattferguson@boisestate.edu
#SBATCH --mail-type=ALL	#This can be set to all, end, or other options
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00         # 24 hours 
#SBATCH --output=logs/gpu-job-%j.o      # output file
#SBATCH --error=logs/gpu-job-%j.e       # error file
#SBATCH --partition=gpuq          # GPU partition
#SBATCH --gres=gpu:1            # single GPU job

ulimit -u 9999
ulimit -s unlimited
ulimit -v unlimited


module load hoomd-blue/gcc/mvapich2/2.1.5

mpirun python hum_gen.py
