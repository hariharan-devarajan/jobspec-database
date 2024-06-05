#!/bin/bash
#SBATCH -A rohit.gajawada
#SBATCH --gres=gpu:1
#SBATCH -n 40
#SBATCH --mincpus=40
#SBATCH --mem-per-cpu=8192
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL

python2 crnn_main.py --random_sample --trainroot='../train_80k/' --valroot='../val_15k/' --cuda --adadelta --batchSize=64

