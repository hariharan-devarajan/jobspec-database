#!/bin/bash
#BSUB -q production
#BSUB -a intelmpi
#BSUB -n 16
#BSUB -R "span[ptile=16]"
#BSUB -m "node20 node21 node22 node01 node02 node04 node05 node06 node07 node08 node09 node10 node11 node12 node14 node15 node16 node17 node18 node19"
#BSUB -J ctrlrun
#BSUB -o run.out
#BSUB -e run.err
#BSUB -W 300:00

#export LD_LIBARAY_PATH=$LD_LIBRARY_PATH:/share/software/intel/mkl/lib/intel64
#source /share/software/intel/bin/compilervars.sh intel64
#source /share/software/intel/mkl/bin/mklvars.sh intel64
#source /share/software/intel/impi/4.0.3.008/bin64/mpivars.sh intel64

export LD_LIBARAY_PATH=$LD_LIBRARY_PATH:/opt/intel/mkl/lib/intel64
source /opt/intel/bin/compilervars.sh intel64
source /opt/intel/mkl/bin/mklvars.sh intel64
source /opt/intel/impi/4.0.3.008/bin64/mpivars.sh intel64

mpirun -genv I_MPI_DEVICE rdma ../build/mitgcmuv

