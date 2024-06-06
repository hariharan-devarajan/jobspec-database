#!/bin/bash
#PBS -l select=1:ncpus=16:mpiprocs=16:ngpus=1:mem=1GB
#PBS -l walltime=05:30:00
  
#PBS -q short_gpuQ
#PBS -N MW_membr_Rec
#PBS -o MW_out
#PBS -e MW_err
  

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
  
/apps/openmpi-3.0.0/bin/mpirun -np 16 /home/giuseppe.gambini/usr/installations/gromacs/bin/gmx_mpi mdrun -s md_meta.tpr -plumed meta_MW.dat -multidir WALKER0 WALKER1 WALKER2 WALKER3
