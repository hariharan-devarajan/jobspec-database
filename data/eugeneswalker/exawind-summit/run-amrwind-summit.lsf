#!/bin/bash
#BSUB -P GEN010
#BSUB -W 0:15
#BSUB -nnodes 2
#BSUB -J AmrWindGPU
#BSUB -o AmrWindGPU.%J
#BUSB -e AmrWindGPU.%J

_SIMG=ecpe4s-exawind-summit-2021-10-05.sif
[[ ! -f "${_SIMG}" && -z ${SIMG+x} ]] && wget "https://cache.e4s.io/exawind/artifacts/${_SIMG}"
SIMG=${SIMG:-$(pwd)/${_SIMG}}

_RUNDIR=${MEMBERWORK}/gen010/exawind-run
RUNDIR=${RUNDIR:-${_RUNDIR}}
[[ ! -d ${RUNDIR} ]] && mkdir -p ${RUNDIR}
cp inputs/amr-wind.input ${RUNDIR}/

module load gcc/9.1.0
module unload darshan-runtime

MPI_HOST=${MPI_ROOT}
MPI_CONTAINER=${MPI_ROOT}

CUDA_HOST=/sw/summit/cuda/11.3.1
CUDA_CONTAINER=/sw/summit/cuda/11.3.1

RUNDIR_HOST=${RUNDIR}
RUNDIR_CONTAINER=/rundir

AMRWIND_CMD_1="amr_wind amr-wind.input amr.n_cell=384 512 512 geometry.prob_hi=300.0 400.0 400.0 time.max_step=10"
AMRWIND_CMD_2="amr_wind amr-wind.input amr.n_cell=384 512 512 geometry.prob_hi=300.0 400.0 400.0 time.max_step=10"
AMRWIND_CMD_1024="amr_wind amr-wind.input amr.n_cell=12288 8192 512 geometry.prob_hi=9600.0 6400.0 400.0 time.max_step=10"

set -x

jsrun \
 --nrs 12 \
 --rs_per_host 6 \
 --tasks_per_rs 1 \
 --cpu_per_rs 1 \
 --gpu_per_rs 1 \
   singularity run \
    --nv \
    --contain \
    --bind /tmp \
    --bind /dev \
    --bind /etc/localtime \
    --bind /etc/hosts \
    --bind /autofs/nccs-svm1_sw \
    --bind /ccs/sw \
    --bind /sw \
    --bind /autofs/nccs-svm1_proj \
    --bind /ccs/proj \
    --bind /sw/summitdev/singularity/98-OLCF.sh:/.singularity.d/env/98-OLCF.sh \
    --bind /etc/libibverbs.d \
    --bind /lib64:/host_lib64 \
    --bind ${RUNDIR_HOST}:${RUNDIR_CONTAINER} \
    --bind ${HOME} \
    --bind ${MPI_HOST} \
    --bind ${CUDA_HOST} \
    --env LD_LIBRARY_PATH=${MPI_CONTAINER}/lib/pami_port \
    ${SIMG} /bin/bash -c "cd ${RUNDIR_CONTAINER} && ${AMRWIND_CMD_2}"
