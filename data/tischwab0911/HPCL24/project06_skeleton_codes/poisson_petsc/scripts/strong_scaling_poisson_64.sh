#!/bin/bash

#SBATCH --nodes=32
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=./../strong/poisson64-10240.out
#SBATCH --error=./../strong/poisson64-10240.err
#SBATCH --time=4:00:00
#SBATCH --constraint=EPYC_7763

# Warmup runs
# mpirun -np 48 ./../poisson -m 1024
# mpirun -np 48 ./../poisson -m 1024
# mpirun -np 48 ./../poisson -m 1024

for np in 4 8 12 16 20 24 28 32; do
    for rep in {1..5}; do
        # mpirun -np $np ./../poisson -m 4096
        # mpirun -np $np ./../poisson -m 6144
        # mpirun -np $np ./../poisson -m 8192
        mpirun -np $np ./../poisson -m 10240
    done
done
