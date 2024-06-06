#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --job-name="4 time,Train tf FRCNN Keras model with only bp data resnet50 as backbone not training shared layers, scheduling on gpu, with input weights as output of last training on cpu"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=r.malhotra.1@student.rug.nl

module load Python/3.6.4-foss-2018a
module load tensorflow/1.5.0-foss-2016a-Python-3.5.2-CUDA-9.1.85
module load cuDNN/7.1.4.18-CUDA-9.0.176


cd /data/s3801128/frcnn
pip install -r requirements.txt --user
pip install tensorflow-gpu --user
python train_frcnn.py --path train_data_bp.txt --input_weight_path only_bp_res50.hdf5 --output_weight_path only_bp_res50_5.hdf5 --config_filename config_only_bp_res50.pickle
