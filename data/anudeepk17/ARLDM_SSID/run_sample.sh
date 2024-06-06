#!/usr/bin/env zsh
#SBATCH --job-name=arldm
#SBATCH --partition=research
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=0-24:00:00
#SBATCH --output="arldm-%j.txt"
#SBATCH -G a100:2
export HYDRA_FULL_ERROR=1
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0+PTX"
export NCCL_DEBUG=WARN
cd $SLURM_SUBMIT_DIR
module load anaconda/full
echo nvidia-smi

bootstrap_conda
conda activate arldm
python3 /srv/home/kumar256/ARLDM/main.py