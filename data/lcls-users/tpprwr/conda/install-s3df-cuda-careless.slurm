#!/bin/bash
#SBATCH -A lcls
#SBATCH --partition=ampere
#SBATCH --job-name=careless-install
#SBATCH --output=careless-install-%j.txt --error=careless-install-%j.txt
#SBATCH --ntasks=1
#SBATCH --gpus a100:1
#SBATCH --cpus-per-task=16

nvidia-smi

source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh

conda create --name careless python=3.9
conda activate careless
pip install --upgrade pip
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install tensorflow==2.12.*

# Install NVCC
conda install -c nvidia cuda-nvcc=11.3.58

# Configure the XLA cuda directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Copy libdevice file to the required path
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/

pip install careless

