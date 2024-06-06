#!/bin/bash

#BSUB -gpu "num=4"
#BSUB -n 2
#BSUB -R "gpuhost span[ptile=1]"
#BSUB -a "docker(registry.gsc.wustl.edu/sleong/osu-micro-benchmark:openmpi-cuda-ofed)"
#BSUB -oo lsf-%J.log

# . /opt/intel/oneapi/setvars.sh
hostlist=$(echo $LSB_HOSTS | tr ' ' '_')
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ibm/lsfsuite/lsf/10.1/linux2.6-glibc2.3-x86_64/lib
export OSU_MPI_DIR=/usr/local/libexec/osu-micro-benchmarks/mpi

# The --mca btl ^openib is to turn off the openib.
# Please do not use IPoIB interfaces for TCP communications.
#    -x UCX_RNDV_SCHEME=get_zcopy \
#    -x UCX_TLS=rc,shm \
mpirun -np 2 -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
    --mca pml ucx \
    --mca btl ^openib \
    -x UCX_MEMTYPE_CACHE=0 \
    -x CUDA_VISIBLE_DEVICES=0 \
    -x UCX_NET_DEVICES=mlx5_0:1 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    --bind-to none \
    -x NCCL_DEBUG=INFO \
    --mca plm_base_verbose 100 \
    --map-by slot \
    --mca orte_base_help_aggregate 0 \
    $TEST -d cuda D D > ./$TEST-$hostlist-$LSB_JOBID.log

