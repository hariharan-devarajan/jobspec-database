#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres gpu:1

# Sherlock module loads
module load boost/1.69.0
module load cmake/3.13.1
module load py-numpy/1.14.3_py27
module load py-pytorch
module load py-scikit-image/0.14.0_py27

export SOFTPATH=/share/software/user/open

export PYCAPATH=$HOME/site-packages/PyCA
export QSPATH=$HOME/gnetree/quicksilver/code/library
export PYTORCHPATH=$SOFTPATH/py-pytorch/1.0.0_py27/lib/python2.7/site-packages
export LD_LIBRARY_PATH=$SOFTPATH/cuda/10.0.130/lib64

export PYTHONPATH=$PWD:$SOFTPATH:$PYCAPATH:$PYTORCHPATH

srun python ./quicksilver/code/applications/qs_predict.py \
    --moving-image ../CUMC_examples/m1.nii \
    --target-image ../CUMC_examples/m2.nii \
    --output-prefix /tmp/
