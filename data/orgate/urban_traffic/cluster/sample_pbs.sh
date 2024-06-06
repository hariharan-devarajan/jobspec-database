#!/bin/bash
# for gpu resource add :gpus=1 or :gpus=2 at the end of following line. Likewise for PHI cards add :mics=1 or :mics=2
#PBS -l nodes=4:ppn=16
#PBS -N dune_grids
# Available Queuename are: 1) nandaq, 2)nandasq, 3)nandagpuq, 4)nandaphiq
#PBS -q nandaq
#PBS -V
#PBS -M alfredajay@imsc.res.in
#PBS -m abe 
#PBS -j oe

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE > hostfile_$PBS_JOBID

#MPI Case
module load intel/2015
module load blas/3.7.0
module load lapack/3.7.0
mpirun -np 64 ~/Dune/2.4.0/dune-user/build-cmake/dune/bidomain/bidomainbath ~/Dune/2.4.0/dune-user/dune/bidomain/parameters.ini >& out_$PBS_JOBID
