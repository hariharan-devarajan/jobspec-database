#!/bin/bash
#SBATCH --account=sc3260
#SBATCH --partition=maxwell
#SBATCH --gres=gpu:4
#SBATCH --mem=10G
#SBATCH --time=20:00:00
#SBATCH --output=multiple_gpu.out

# Set up environment
module load GCC Singularity git

# Clone tensorflow repo to run a tutorial
# git clone https://github.com/tensorflow/models.git
singularity exec --nv docker://tensorflow/tensorflow:latest-gpu \
    python ../tf_cnn_benchmarks.py --num_gpus=4 --batch_size=32 --model=resnet50 --variable_update=parameter_server
