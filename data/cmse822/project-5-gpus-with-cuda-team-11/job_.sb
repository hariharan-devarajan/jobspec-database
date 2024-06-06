#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --time=00:05:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --ntasks=1                # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1         # number of CPUs (or cores) per task (same as -c)
#SBATCH --gpus=v100:1
#SBATCH --mem-per-gpu=500MB
########## Command Lines to Run ##########

#echo "$SLURM_ARRAY_TASK_ID"


RANDOM=$$



module load NVHPC/21.9-GCCcore-10.3.0-CUDA-11.4
nvcc final_diffusion_code.cu -o final_diffusion -O3
./final_diffusion.c
