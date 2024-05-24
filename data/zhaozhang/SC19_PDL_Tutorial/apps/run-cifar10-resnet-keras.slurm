#!/bin/bash
#SBATCH -J Cifar10-1node    # job name
#SBATCH -o Cifar10-1node.out         # output and error file name (%j expands to jobID)
#SBATCH -N 1              # total number of nodes
#SBATCH -n 1
#SBATCH -p normal           # queue (partition) -- normal, development, etc.
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 4 hours
#SBATCH -A Intel-TensorFlow   #Intel-TensorFlow    #TG-CCR150011    # project name

export OMP_NUM_THREADS=64
export KMP_BLOCKTIME=0
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
module load phdf5

mkdir -p /dev/shm/keras
cp /home1/apps/keras/data/datasets.tar /dev/shm/keras/datasets.tar 
tar xf /dev/shm/keras/datasets.tar -C /dev/shm/keras

ibrun -np 1 python cifar10_resnet_keras.py
