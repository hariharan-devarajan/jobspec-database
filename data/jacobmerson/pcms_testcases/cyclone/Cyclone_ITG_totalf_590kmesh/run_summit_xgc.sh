#!/bin/bash
# Begin LSF Directives
#BSUB -P FUS123
#BSUB -W 2:00
#BSUB -nnodes 4
#BSUB -J XGC
#BSUB -o XGC.%J
#BSUB -e XGC.%J

module load nvhpc/21.7
module load spectrum-mpi/10.4.0.3-20210112
module load netlib-lapack/3.9.1
module load hypre/2.22.0-cpu
module load fftw/3.3.9
module load hdf5/1.10.7
module load cmake/3.20.2
module load libfabric/1.12.1-sysrdma

export OMP_NUM_THREADS=14

date
jsrun -n 24 -r 6 -a 1 -g 1 -c 7 -b rs ./xgc-es-cpp-gpu
date
