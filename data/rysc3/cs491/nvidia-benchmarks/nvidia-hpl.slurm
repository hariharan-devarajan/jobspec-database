#!/bin/bash
#SBATCH --job-name nvidia-hpl
#SBATCH -N 1
#SBATCH -p condo
#SBATCH --ntasks-per-node 4
#SBATCH -G 4  # Request all four MIG instances
#SBATCH --mem-per-gpu 10GB  # Adjust memory usage per GPU instance
#SBATCH --mail-type all
#SBATCH --mail-user rscherbarth@unm.edu

module load singularity
module load openmpi

DATESTRING=$(date "+%Y-%m-%dT%H:%M:%S")
CONT='/users/rdscher/CLASS/cs491/nvidia-benchmarks/hpc-benchmarks:24.03.sif'
MOUNT="/users/rdscher/CLASS/cs491/nvidia-benchmarks"

echo "------- Node --------"
hostname
echo "---------------------"

echo "Running on hosts: $(echo $(scontrol show hostname))"
echo "$DATESTRING"

# Specify GPU instance IDs for the MIG devices
# Adjust the script to utilize all four MIG instances
srun --mpi=pmi2 singularity run --nv --bind .:/my-dat-files hpc-benchmarks:24.03.sif ./hpl.sh --dat HPL-4GPUs.dat

echo "Done"
echo "$DATESTRING"

