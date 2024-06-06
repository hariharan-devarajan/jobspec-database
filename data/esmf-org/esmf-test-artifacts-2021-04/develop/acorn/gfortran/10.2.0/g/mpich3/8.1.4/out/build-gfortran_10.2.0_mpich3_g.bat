#!/bin/sh -l
#PBS -N build-gfortran_10.2.0_mpich3_g.bat
#PBS -l walltime=1:00:00
#PBS -l walltime=1:00:00
#PBS -q workq
#PBS -A emc
#PBS -l select=1:ncpus=128:mpiprocs=128
JOBID="`echo $PBS_JOBID | cut -d. -f1`"

cd /lfs/h1/emc/ptmp/Mark.Potts/gfortran_10.2.0_mpich3_g_develop

module unload PrgEnv-cray PrgEnv-intel

module load PrgEnv-gnu cray-pals
module load gcc/10.2.0 cray-mpich/8.1.4 cray-netcdf/4.7.4.3
module load cray-hdf5/1.12.0.3 
module list >& module-build.log

set -x
export ESMF_NETCDF=nc-config

export ESMF_OS=Linux
export ESMF_CXXCOMPILER=CC
export ESMF_F90COMPILER=ftn
export ESMF_CXXLINKER=CC
export ESMF_F90LINKER=ftn
export ESMF_MPIRUN=mpirun.unicos
export ESMF_F90COMPILEOPTS="-fallow-argument-mismatch -fallow-invalid-boz"
export ESMF_NFCONFIG=nf-config
export ESMF_CXXLINKOPTS="-fPIC -lnetcdff -lnetcdff"
sed -i 's/aprun/mpiexec/' scripts/mpirun.unicos
export ESMF_DIR=/lfs/h1/emc/ptmp/Mark.Potts/gfortran_10.2.0_mpich3_g_develop
export ESMF_COMPILER=gfortran
export ESMF_COMM=mpich3
export ESMF_BOPT='g'
export ESMF_TESTEXHAUSTIVE='ON'
export ESMF_TESTWITHTHREADS='ON'
make -j 128 clean 2>&1| tee clean_$JOBID.log 
make -j 128 2>&1| tee build_$JOBID.log

ssh alogin01 /lfs/h1/emc/ptmp/Mark.Potts/gfortran_10.2.0_mpich3_g_develop/getres-build.sh
