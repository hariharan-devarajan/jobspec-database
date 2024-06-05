#!/bin/bash
#SBATCH --job-name=project
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=6epochtest.txt
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --mail-type=END
#SBATCH --mail-user=yl8798@nyu.edu

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate key
module load python/intel/3.8.6
cd project

python train_time.py --upscale_factor 4 --cuda --epochs 6 --bs 32
python train_time.py --upscale_factor 4 --cuda --epochs 6 --bs 64
python train_time.py --upscale_factor 4 --cuda --epochs 6 --bs 64 --lr 0.0002
