#!/bin/bash
#SBATCH --job-name benchmarks
#SBATCH --mail-user jastewart@unm.edu
#SBATCH --mail-type FAIL,TIME_LIMIT
#SBATCH --output run_hopper.out
#SBATCH --error run_hopper.err
#SBATCH --ntasks 16
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks-per-node 16
#SBATCH --partition general
#SBATCH --exclusive
#SBATCH --time 4:00:00
#SBATCH --distribution block:block

spack load openmpi/nb2qima72b5usivgbcdbkwn5ivmcwlxk
mpirun -n 1 all_to_all 0
mpirun -n 2 all_to_all 0
mpirun -n 4 all_to_all 0
mpirun -n 6 all_to_all 0
mpirun -n 8 all_to_all 0
mpirun -n 16 all_to_all 0
mpirun -n 24 --oversubscribe all_to_all 1
mpirun -n 32 --oversubscribe all_to_all 1
mpirun -n 64 --oversubscribe all_to_all 1
mpirun -n 128 --oversubscribe all_to_all 1

mpirun -n 1 count_primes 0
mpirun -n 2 count_primes 0
mpirun -n 4 count_primes 0
mpirun -n 6 count_primes 0
mpirun -n 8 count_primes 0
mpirun -n 16 count_primes 0
mpirun -n 24 --oversubscribe count_primes 1
mpirun -n 32 --oversubscribe count_primes 1
mpirun -n 64 --oversubscribe count_primes 1
mpirun -n 128 --oversubscribe count_primes 1
