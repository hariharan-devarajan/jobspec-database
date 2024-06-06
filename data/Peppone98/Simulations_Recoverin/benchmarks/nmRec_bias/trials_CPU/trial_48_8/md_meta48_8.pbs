#!/bin/bash
#PBS -l select=1:ncpus=48:mpiprocs=8:mem=1GB
#PBS -l walltime=00:05:00
  
#PBS -q short_cpuQ
#PBS -N md_48_8
#PBS -o md_48_8_out
#PBS -e md_48_8_err
  

module load gcc91
module load openmpi-3.0.0
module load BLAS
module load gsl-2.5
module load lapack-3.7.0
  
module load cuda-11.3

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/giuseppe.gambini/usr/installations/plumed/lib
source /home/giuseppe.gambini/usr/src/gmx_plumed.sh

export OMP_NUM_THREADS=6
  
/apps/openmpi-3.0.0/bin/mpirun -np 8 /home/giuseppe.gambini/usr/installations/gromacs/bin/gmx_mpi mdrun -s ../../md_meta.tpr -plumed meta.dat
