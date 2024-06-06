#!/bin/bash
#PBS -l select=1:ncpus=48:mpiprocs=6:mem=1GB
#PBS -l walltime=00:10:00
  
#PBS -q short_cpuQ
#PBS -N Ad_Geom_
#PBS -o Ad_Geom_nmRec_out
#PBS -e Ad_Geom_nmRec_err
  

module load gcc91
module load openmpi-3.0.0
module load BLAS
module load gsl-2.5
module load lapack-3.7.0
  
module load cuda-11.3

cd $PBS_O_WORKDIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/giuseppe.gambini/usr/installations/plumed/lib

source /home/giuseppe.gambini/usr/src/gmx_plumed.sh

export OMP_NUM_THREADS=8
  
/apps/openmpi-3.0.0/bin/mpirun -np 6 /home/giuseppe.gambini/usr/installations/gromacs/bin/gmx_mpi mdrun -s nmRec.tpr -plumed meta_AG.dat