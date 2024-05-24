#!/bin/bash
#SBATCH --job-name=distributed_training
#SBATCH --partition=gpu

# 2 nodes

#SBATCH --nodes=2
 
# two GPUs

#SBATCH --gres=gpu:2

# I have set the following to some random values

#SBATCH --ntasks=8                      # Number of MPI ranks

#SBATCH --cpus-per-task=8               # Number of cores per MPI rank 

#SBATCH --ntasks-per-node=4             # How many tasks on each node

#SBATCH --ntasks-per-socket=2           # How many tasks on each CPU or socket

#SBATCH --mem-per-cpu=100mb             # Memory per core

# set to 16 but should be changed to match our specifications

#OR SWAP mem per cpu for #SBATCH --mem=16G

# maximum run time (set to 1 hour just because)

#SBATCH --time=01:00:00

# loading modules

module load cuda/11.0
module load cudnn/8.0.2

# same as the bash script from here

# replace with ipynb OR download the python script from colab and replace

training="train.py" 

# or uncomment to test if this works

# jupyter nbconvert --to script Train_YOLOv5_ipynb.ipynb

# checking existence

if [ -f "$training" ];
then
    echo "running"

    # in the example I based this off of, they used mpi (message passing interface) + srun is used for slurm
    
    srun --mpi=pmix_v3 python3 "$training"
    
    echo "done"
else
    echo "does not exist"
fi