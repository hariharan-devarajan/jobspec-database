#!/bin/bash

# Submit to the tcb 1ition
#SBATCH -p lindahl1,lindahl2,lindahl3

# The name of the job in the queue
#SBATCH -J jobname
# wall-clock time given to this job
#SBATCH -t 24:00:00

# Number of nodes and number of MPI processes per node
#SBATCH -N 1 
# Request a GPU node and 4 GPUs (per node)
#SBATCH -C gpu --gres=gpu:4

# Output file names for stdout and stderr
#SBATCH -e job-%j.err -o job-%j.out

# Receive e-mails when your job starts and ends

#SBATCH -d singleton

# The actual script starts here
module unload gromacs
module switch gromacs/2023 gromacs=gmx_mpi
module switch cuda/11.8
module unload openmpi
module load openmpi

srun -n 4 gmx_mpi mdrun -deffnm md -cpi md -multidir rep1 rep2 rep3 rep4 -ntomp $((SLURM_JOB_CPUS_PER_NODE/4)) -maxh 23
