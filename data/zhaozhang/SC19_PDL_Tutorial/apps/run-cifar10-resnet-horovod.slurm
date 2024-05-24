#!/bin/bash
#SBATCH -J Cifar10-4node    # job name
#SBATCH -o Cifar10-4node.out         # output and error file name (%j expands to jobID)
#SBATCH -N 4              # total number of nodes
#SBATCH -n 4
#SBATCH -p normal           # queue (partition) -- normal, development, etc.
#SBATCH -t 01:00:00        # run time (hh:mm:ss) - 4 hours
#SBATCH -A TACC-DIC    #TG-CCR150011    # project name

export OMP_NUM_THREADS=64
export KMP_BLOCKTIME=0
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
module load phdf5

ibrun -np 4 mkdir /dev/shm/keras
/home1/apps/dl-tools/bin/broadcast-mpi.sh /home1/apps/keras/data/datasets.tar /dev/shm/keras/datasets.tar 4
ibrun -np 4 tar xf /dev/shm/keras/datasets.tar -C /dev/shm/keras

ibrun -np 4 python cifar10_resnet_horovod.py
