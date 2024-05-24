#!/bin/bash

#SBATCH --job-name=plumed
#SBATCH --get-user-env
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

module load intel
module load intel-mkl
module load intel-mpi
module load lammps

LAMMPS=lmp_mpi

cd example_natasha

srun --time=24:00:00 --hint=nomultithread --exclusive -n ${SLURM_NTASKS} ${LAMMPS} -i in.lmp > log.lammps
