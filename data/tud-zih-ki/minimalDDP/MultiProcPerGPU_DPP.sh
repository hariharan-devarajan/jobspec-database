#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH -J TestDDP_GPUBind_MPIDatloader
#SBATCH --output=Test-R-%j-%x.log
#SBATCH --gres=gpu:8
#SBATCH -p alpha,alpha-interactive
#SBATCH --mem=0
#SBATCH -N 3
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=single:3
#SBATCH --constraint=fs_beegfs
#SBATCH --hint=nomultithread
## 
module purge
ml modenv/hiera 
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml imkl/2022.0.1
ml CUDA/11.7.0
ml cuDNN/8.4.1.50-CUDA-11.7.0
ml NCCL/2.12.12-CUDA-11.7.0
ml Python/3.9.6-bare
source venv

srun --distribution=plane=3 python3 -u MultiProcPerGPU_DPPMPI.py
