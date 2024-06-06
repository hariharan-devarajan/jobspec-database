#!/bin/bash
# This is a template for a Job Array
# A Job Array runs multiple instances of the same job with different values of PBS_ARRAY_INDEX

# You should use a project id, either the 8 digit code 
# of your project or "Personal"
#PBS -P Personal

# Range of array indices
#PBS -J 0-2

# Name of job
#PBS -N array

# Resources per subjob
#PBS -l select=1:ncpus=2:mpiprocs=2:ompthreads=1:mem=8gb

# Required elapsed time per subjob
#PBS -l walltime=0:10:00

# This option merges standard output and standard error in the same file
#PBS -j oe

# Load any modules which are not in your login profile
# The composerxe module has the Intel compiler and Intel MPI
module load intelcc intelmpi

## # Print environment
## ulimit -a
## env
## qstat -f $PBS_JOBID

# Change directory
cd "$PBS_O_WORKDIR"

echo This is PBS array index: $PBS_ARRAY_INDEX

# In this example the executable uses a different input file for each subjob
mpirun ./a.out input.$PBS_ARRAY_INDEX
