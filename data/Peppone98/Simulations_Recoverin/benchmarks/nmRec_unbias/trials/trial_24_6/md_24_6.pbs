#!/bin/bash
#PBS -l select=1:ncpus=24:mpiprocs=6:mem=1GB
#PBS -l walltime=00:05:00
  
#PBS -q short_cpuQ
#PBS -N md_24_6
#PBS -o md_24_6_out
#PBS -e md_24_6_err
  

module load gcc91
module load openmpi-3.0.0
module load BLAS
module load gsl-2.5
module load lapack-3.7.0
  
module load cuda-11.3

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/giuseppe.gambini/usr/installations/plumed/lib
source /home/giuseppe.gambini/usr/src/gmx_plumed.sh

export OMP_NUM_THREADS=4
  
/apps/openmpi-3.0.0/bin/mpirun -np 6 /home/giuseppe.gambini/usr/installations/gromacs/bin/gmx_mpi mdrun -s ../../md.tpr
