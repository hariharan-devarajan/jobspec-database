#!/bin/bash

#SBATCH -N 1                        # number of compute nodes
#SBATCH -n 1                        # number of tasks your job will spawn
#SBATCH --mem=32G                   # amount of RAM requested in GiB (2^40)
#SBATCH -p sulcgpu1                # Use gpu partition
#SBATCH -q wildfire                 # Run job under wildfire QOS queue
#SBATCH --gres=gpu:A6000:1           # Request two GPUs
#SBATCH -t 0-11:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when a job starts, stops, or fails
#SBATCH --mail-user=kkasodek@asu.edu # send-to address
...
nvidia-smi # Useful for seeing GPU status and activity 
python train.py
...