#!/bin/bash
#SBATCH --job-name=mnist_2GPUs       # Job name
#SBATCH --mail-type=ALL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=email@ufl.edu    # Where to send mail	
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=64gb                   # Job memory request
#SBATCH --partition=gpu              # Use the gpu partition
#SBATCH --gpus=a100:2                # Request 2 A100 gpus
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --output=mnist_batch_%j.log     # Standard output and error log

pwd; hostname; date           # Print some useful info

module load tensorflow/2.7.0        # Be sure to load the tensorflow module

echo "Running mnist script"

python distributed_mnist.py  # Run the script

date