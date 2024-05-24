#!/bin/bash -l
# Set number of tasks to run
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 2
# Set the number of CPU cores for each task
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
# Walltime format hh:mm:ss
#SBATCH --time=72:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err
FILES=(/scratch/jhh508/stable-diffusion-2/*)


# **** Put all #SBATCH directives above this line! ****
# **** Otherwise they will not be effective! ****

module purge

pwd

cd /scratch/jhh508/stable-diffusion-2/

eval "$(conda shell.bash hook)"

conda init bash

conda activate stable-diff

module load gcc

python scripts/guidance_inference_combo_neg.py --csv webDiffusion.csv --config configs/stable-diffusion/v2-inference.yaml --ckpt 512-base-ema.ckpt --fixed_code --negFile
