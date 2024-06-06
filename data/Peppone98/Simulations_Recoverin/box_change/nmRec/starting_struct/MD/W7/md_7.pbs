#!/bin/bash
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=1GB
#PBS -l walltime=01:00:00
  
#PBS -q short_gpuQ
#PBS -N md_7
#PBS -o md_7_out
#PBS -e md_7_err
  

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
  
/apps/openmpi-3.0.0/bin/mpirun -np 1 /home/giuseppe.gambini/usr/installations/gromacs/bin/gmx_mpi mdrun -s md_7.tpr -nb gpu -pme auto
