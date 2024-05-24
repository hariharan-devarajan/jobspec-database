#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=0-120:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=20G
#SBATCH --output /home/ivrik/sounds/modelo.log
#SBATCH -q gpgpuresplat
#SBATCH -A punim1410
#SBATCH --partition=gpgpu
#SBATCH --qos=gpgpuresplat
## Use an account that has GPGPU access


# Load required modules
module load gcccore/8.3.0
module load python/3.7.4

module purge
module load fosscuda/2019b

module load tensorflow/2.1.0-python-3.7.4
#module load Python/2.7.10-goolf-2015a
module load  python/3.7.4
##/3.6.4-intel-2018.u4 

# Launch python code
cd ~/sounds
python model.py  ${OPTS} &> model${VAR}.log


