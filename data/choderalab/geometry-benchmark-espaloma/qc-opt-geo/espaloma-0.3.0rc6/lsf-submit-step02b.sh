#!/bin/bash
#BSUB -P "qc-opt-geo"
#BSUB -J "array[1-250]"
#BSUB -n 1
#BSUB -R rusage[mem=8]
#BSUB -R span[hosts=1]
#BSUB -q gpuqueue
#BSUB -W 6:00
#BSUB -L /bin/bash
#BSUB -o out-%J.out
#BSUB -e out-%J.err

source ~/.bashrc

# Set the output and error output paths.

mkdir -p 02-outputs

model="espaloma-0.3.0rc6"
fftype="espaloma"
ffpath="/home/takabak/.espaloma/${model}.pt"

conda activate qc-opt-geo
python 02-b-minimize.py -i ../espaloma-0.3.0rc1-pavan/02-chunks/01-processed-qm-${LSB_JOBINDEX}.sdf -fftype ${fftype} -ffpath ${ffpath} -o 02-outputs/${model}-${LSB_JOBINDEX}.sdf
