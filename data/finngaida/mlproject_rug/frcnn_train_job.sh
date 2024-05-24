#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --mem=32GB
#SBATCH --job-name="Train th FRCNN Keras model with some POC data"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=f.m.gaida@student.rug.nl

module load Python/3.6.4-foss-2018a
module load tensorflow/1.5.0-foss-2016a-Python-3.5.2-CUDA-9.1.85
module load OpenCV/3.1.0-foss-2016a-Python-3.5.1

cd /data/s3838730/ml/keras-frcnn
pip install tensorflow --user
pip install -r requirements.txt --user
python train_frcnn.py -p VOCdevkit --num_epochs 1000
