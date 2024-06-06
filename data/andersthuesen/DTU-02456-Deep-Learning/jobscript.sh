#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ocr
#BSUB -n 8
#BSUB -R "span[block=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -W 20:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u andersbthuesen@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o logs/run3-%J.out
#BSUB -e logs/run3-%J.err

nvidia-smi

PATH=~/miniconda3/bin:$PATH

./train.py