#!/bin/bash
#SBATCH --job-name=stream4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --output=build/stream_final.out

module load compilers/armclang/24.04
module load libraries/openmpi/5.0.3/armclang-24.04

# Turns out the below run is pretty bad
# mpirun -np 64 --map-by ppr:16:node --bind-to core ./STREAM/stream

# Better performance but its running one process per node
# Bit of reading and ive found that it measures bandwith on a SINGLE node
# The benchmark runs independently on each node, pointless exercise to run on 4 nodes
# mpirun -np 4 --map-by ppr:1:node ./STREAM/stream

# This runs one instance


export OMP_NUM_THREADS=16

export OMP_PROC_BIND=spread


# these dont run on the whole CPU, for some reason it restricts to single core
# mpirun -np 1 ./STREAM/stream
# mpirun -np 1 --map-by ppr:1:node ./STREAM/stream


# This runs on one node and all 16 cores
srun -n 1 ./STREAM/stream

