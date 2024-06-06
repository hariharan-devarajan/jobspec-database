#!/bin/bash
#BSUB -P "qc-opt-geo"
#BSUB -J "array[1-250]"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -W 8:00
#BSUB -L /bin/bash
#BSUB -o out-%J.out
#BSUB -e out-%J.err

source ~/.bashrc

# Set the output and error output paths.

mkdir -p 02-outputs

model="gaff-2"
fftype="gaff"
ffpath="gaff-2.11"

conda activate qc-opt-geo
python 02-b-minimize.py -i ../espaloma-0.3.0rc1-pavan/02-chunks/01-processed-qm-${LSB_JOBINDEX}.sdf -fftype ${fftype} -ffpath ${ffpath} -o 02-outputs/${model}-${LSB_JOBINDEX}.sdf
