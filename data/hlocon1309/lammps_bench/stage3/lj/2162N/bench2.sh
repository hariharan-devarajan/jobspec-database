#!/bin/sh
#SBATCH -N 2
#SBATCH -n 16
#SBATCH -p normal
#SBATCH --nodelist=cn-[017,018]
#SBATCH --gpus-per-node=2
#SBATCH -A proj_1371
#SBATCH --constraint=type_b

module load openmpi/4.1.4
module load CUDA/11.7
module load lammps/2022jun23_update1

srun --mpi=pmix_v2 lmp -i in.lj -sf gpu -pk gpu 2
