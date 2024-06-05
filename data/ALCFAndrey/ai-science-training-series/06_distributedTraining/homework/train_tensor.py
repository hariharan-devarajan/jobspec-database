#!/bin/bash -l
#COBALT -t 20
#COBALT -q training-gpu
#COBALT -n 1
#COBALT -A ALCFAITP

#  #########3#COBALT --attrs filesystems=home:eagle

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
mpirun -np 1 python tensorflow2_mnist_hvd.py --metrics 1
mpirun -np 2 python tensorflow2_mnist_hvd.py --metrics 2
mpirun -np 4 python tensorflow2_mnist_hvd.py --metrics 4
mpirun -np 8 python tensorflow2_mnist_hvd.py --metrics 8
#mpirun -np 16 python tensorflow2_mnist_hvd.py --metrics 16 #Not available
