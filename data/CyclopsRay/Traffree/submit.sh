#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=24:00:00 -N 16

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (4GB) (CPU RAM):
# Specify a job name:
#SBATCH -J Preprocess

# Specify an output file
#SBATCH -o preprocess.out


# Set up the environment by loading modules
module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
module load tree

cd finalProject
#conda create -n finalProject python=3.10
conda activate finalProject
#pip install --upgrade pip
#pip install Pillow
pip install tqdm
#pip install -r ~/wavelet-transformer/env/requirement.txt
#conda info --envs
#pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
#module load gcc/10.2
#pip install wandb
#cd ~/wavelet-transformer
pwd
#python preprocess.py
python create_task.py
#export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
#unzip Real_Train.zip
#unzip Unreal_Train.zip
#
##python3 setup.py develop --no_cuda_ext
#
#torchrun --standalone --nproc_per_node=2  basicsr/train.py -opt options/train/SIDD/Wavelet-width32.yml --launcher pytorch --batch_size=8
