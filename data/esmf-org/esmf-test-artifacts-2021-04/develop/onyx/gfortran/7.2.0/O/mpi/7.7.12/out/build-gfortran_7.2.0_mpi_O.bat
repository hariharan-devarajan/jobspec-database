#!/bin/sh -l
#PBS -N build-gfortran_7.2.0_mpi_O.bat
#PBS -l walltime=1:00:00
#PBS -l walltime=1:00:00
#PBS -q standard
#PBS -A NRLMR03795024
#PBS -l select=1:ncpus=44:mpiprocs=44
JOBID="`echo $PBS_JOBID | cut -d. -f1`"

cd /p/work/mpotts/gfortran_7.2.0_mpi_O_develop

module unload PrgEnv-cray PrgEnv-intel

module load PrgEnv-gnu
export ESMF_F90COMPILER=ftn
module load gcc/7.2.0 cray-mpich/7.7.12 netcdf/gcc-7.3.0/4.6.2
module list >& module-build.log

set -x
export ESMF_NETCDF=nc-config

export ESMF_DIR=/p/work/mpotts/gfortran_7.2.0_mpi_O_develop
export ESMF_COMPILER=gfortran
export ESMF_COMM=mpi
export ESMF_BOPT='O'
export ESMF_TESTEXHAUSTIVE='ON'
export ESMF_TESTWITHTHREADS='ON'
make -j 44 clean 2>&1| tee clean_$JOBID.log 
make -j 44 2>&1| tee build_$JOBID.log

ssh onyx /p/work/mpotts/gfortran_7.2.0_mpi_O_develop/getres-build.sh
