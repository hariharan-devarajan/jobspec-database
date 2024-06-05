#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --job-name=test

#SBATCH --time=60:00:00 # time

#SBATCH --ntasks=1 # number of processor cores (i.e. tasks)

#SBATCH --nodes=1 # number of nodes

#SBATCH --mem-per-cpu=50G # memory per CPU core

#SBATCH --gres=gpu:1 # request 1 gpu 

#SBATCH --cpus-per-task=4
#SBATCH --partition=wsu_gen_gpu.q
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load Python/3.9.6-GCCcore-11.2.0
source /home/p793x363/Documents/test/bin/activate
python Wav2vec2.py  #run python script