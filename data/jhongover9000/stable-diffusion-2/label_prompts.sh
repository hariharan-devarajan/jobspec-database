#!/bin/bash -l
# Set number of tasks to run
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH -n 5
# Set the number of CPU cores for each task
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
# Walltime format hh:mm:ss
#SBATCH --time=72:00:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err
FILES=(/scratch/jhh508/stable-diffusion-2/*)


# **** Put all #SBATCH directives above this line! ****
# **** Otherwise they will not be effective! ****

module purge

cd /scratch/jhh508/stable-diffusion-2/prompt-labeling/

pwd

eval "$(conda shell.bash hook)"

conda init bash

conda activate stable-diff

module load gcc

echo loaded

chmod +x promptLabeler.py

python promptLabeler.py promptList_full.txt labelList_full_v2.txt
