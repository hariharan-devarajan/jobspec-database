#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --job-name=VanillaDQN

module load Python/3.7.4-GCCcore-8.3.0 
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4
module load OpenAI-Gym/0.17.1-foss-2019b-Python-3.7.4

source /data/$USER/.envs/pyenv37/bin/activate

echo Starting Python program

python /data/s3972445/.envs/pyenv37/dreaming-rl/main_peregrine.py
 







