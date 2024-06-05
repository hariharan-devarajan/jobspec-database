#!/bin/bash

#SBATCH --job-name=DNA
#SBATCH --account=nn4654k
#SBATCH --partition=normal
#SBATCH --time=0-0:30:00
#SBATCH --mem-per-cpu=3000M 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --qos=devel

## Set up job environment
#source /cluster/bin/jobsetup

# Set up node file for namd run :
module load PETSc/3.4.4
make clean
make
mpiexec -n 2 test


exit 0







#######

