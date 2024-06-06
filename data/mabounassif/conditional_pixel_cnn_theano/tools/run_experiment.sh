#!/bin/bash
#PBS -l walltime=30:00:00
#PBS -r n
#PBS -e $HOME/class_project/outputs/error.txt
#PBS -o $HOME/class_project/outputs/output.txt

module load python/3.5.1
module load openblas/0.2.18
module load theano/python3.5/0.7.0
module add CUDA

cd $HOME/class_project

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python tools/test_gpu.py
