#!/bin/sh
### General options
### –- specify queue --
PBS -q gputitanxpascal
### -- set the job Name --
PBS -N cdio_calc
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
PBS -l walltime=04:00:00
# request 5GB of memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
PBS -M olivernybroe@gmail.com
PBS -m abe
# -- end of LSF options --

nvidia-smi
# Load modules
module load tensorflow/1.5-gpu-python-3.6.2
module load python3/3.6.2
module load opencv/3.3.1-python-3.6.2
module load cython/0.28.1-python-3.6.2


cd darkflow
flow --model cfg/tiny-yolo-voc-2c.cfg --load -1  --train --annotation ./train/Annotations/ --dataset ./train/Images --gpu 1.0