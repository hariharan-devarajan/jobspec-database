#!/bin/bash
#BSUB -P FUS157
#BSUB -W 0:60
#BSUB -nnodes 1
#BSUB -J Cyclone-590k
#BSUB -o Cyclone-590k.%J
#BSUB -e Cyclone-590k.%J

module load gcc/12.1.0
module load cuda/12.2.0

date
jsrun -n 6 -a 1 -c 1 -g 1 --smpiargs "-gpu" \
./XGCm --kokkos-num-threads=1 590kmesh.osh 590kmesh_6.cpn \
1 1 bfs bfs 0 0 0 3 input_xgcm petsc petsc_xgcm_cpu.rc \
-use_gpu_aware_mpi 0
date
