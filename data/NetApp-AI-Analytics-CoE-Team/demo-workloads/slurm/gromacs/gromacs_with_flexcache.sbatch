#!/bin/bash
#SBATCH --job-name=gromacs-threadmpi
#SBATCH --exclusive
#SBATCH --output=/shared-non-cache/logs/%x_%j.out
#SBATCH -N 4
#SBATCH -n 64
NTOMP=1

mkdir -p /shared-non-cache/output/jobs/${SLURM_JOBID}
cd /shared-non-cache/output/jobs/${SLURM_JOBID}

spack load gromacs
module load openmpi

set -x
time mpirun -np ${SLURM_NTASKS} gmx_mpi mdrun -ntomp ${NTOMP} -s /shared-cache/input/gromacs/benchRIB.tpr -resethway

