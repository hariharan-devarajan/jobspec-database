#!/bin/bash
#PBS -l select=1:ncpus=8:mpiprocs=8:ngpus=1:mem=1GB
#PBS -l walltime=00:30:00
  
#PBS -q short_gpuQ
#PBS -N MW_nmRec
#PBS -o walk_out
#PBS -e walk_err
  

module load gcc91
module load openmpi-3.0.0
module load BLAS
module load gsl-2.5
module load lapack-3.7.0
  
module load cuda-11.3

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/giuseppe.gambini/usr/installations/plumed/lib

source /home/giuseppe.gambini/usr/src/gmx_plumed.sh

export OMP_NUM_THREADS=1

/apps/openmpi-3.0.0/bin/mpirun -np 8 /home/giuseppe.gambini/usr/installations/gromacs/bin/gmx_mpi mdrun -s md_Meta.tpr -plumed Walkers.dat -multidir WALKER0 WALKER1 WALKER2 WALKER3 WALKER4 WALKER5 WALKER6 WALKER7
