#!/bin/bash

#BSUB -P CSC340 
##    BSUB -W 06:00
#BSUB -W 00:10
#BSUB -nnodes 1
##   BSUB -alloc_flags smt4
#BSUB -J r_pelelm
#BSUB -o rout_pelelm_1gpu_o.%J
#BSUB -e rout_pelelm_1gpu_e.%J

module load gcc
module load python
module load cuda
module load cmake

export LD_LIBRARY_PATH=/ccs/home/cyrush/WORKSCRATCH/2021_11_cyrush_pelelm/PelePhysics/ThirdParty/INSTALL/gcc.CUDA/lib/:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
jsrun -r 6 -a 1 -g 1 -c 7 --bind=packed:7 \
  ./PeleLM3d.gnu.MPI.CUDA.ex input_1GPU

find . -type f -exec chmod g+rw {} \;
find . -type d -exec chmod g+rwx {} \;
