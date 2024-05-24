#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=l.n.faber@student.rug.nl
#SBATCH --job-name=yolo_small
#SBATCH --output=keras_tiny.out

module load Python/3.6.4-intel-2018a
module load CUDA/9.1.85
module load tensorflow/1.5.0-foss-2016a-Python-3.5.2-CUDA-9.1.85
python train2.py
