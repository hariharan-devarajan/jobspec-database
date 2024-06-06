#!/bin/bash
#SBATCH --job-name=ring_shift
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --constraint=intel18

# Load necessary modules
module purge
module load gcc/7.3.0-2.30 openmpi hdf5 python git

# Powers of 2 for message sizes from 2 bytes to 4096 bytes
for size in 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131076; do
    # Adjust 'N' according to your cluster's specifics for separate nodes
    for procs in 2 4 8 16 32 64; do
        echo "Running with $procs processes and message size $size"
        srun -n $procs ./part3q2.o $size
    done
done
