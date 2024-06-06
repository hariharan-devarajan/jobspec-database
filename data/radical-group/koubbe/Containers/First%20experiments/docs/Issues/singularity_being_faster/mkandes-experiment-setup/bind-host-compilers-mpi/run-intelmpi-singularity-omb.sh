#!/usr/bin/env bash

#SBATCH --job-name=intelmpi-singularity-omb-latency
#SBATCH --account=use300
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=intelmpi-singularity-omb-latency.o%j.%N

declare -xr COMPILER_MODULE='intel/2018.1.163'
declare -xr MPI_MODULE='intelmpi/2018.1.163'

declare -xr OMB_VERSION='5.6.2'
declare -xr OMB_BUILD='mpi'
declare -xr OMB_ROOT_DIR="${PWD}"
declare -xr OMB_ROOT_URL='http://mvapich.cse.ohio-state.edu/download/mvapich'
declare -xr OMB_INSTALL_DIR="${OMB_ROOT_DIR}/${OMB_VERSION}/${COMPILER_MODULE}/${MPI_MODULE}/${OMB_BUILD}"

module purge
module load "${COMPILER_MODULE}"
module load "${MPI_MODULE}"
module list
export SINGULARITYENV_PREPEND_PATH=/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/mpi/intel64/bin:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/bin/intel64
export SINGULARITYENV_LD_LIBRARY_PATH=/etc/libibverbs.d:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/compiler/lib/intel64_lin:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/ipp/../compiler/lib/intel64:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/ipp/lib/intel64:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64:/opt/intel/2018.1.163/compilers_and_libraries_2018.1.163/linux/tbb/lib/intel64/gcc4.4
printenv

time -p mpirun singularity exec --bind /etc,/opt,/oasis,/scratch centos_latest.sif "${OMB_INSTALL_DIR}/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency"
