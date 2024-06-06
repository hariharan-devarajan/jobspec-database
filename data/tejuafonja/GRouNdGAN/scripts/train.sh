#!/bin/bash

#SBATCH --job-name=groundgan
#SBATCH --time=24:00:00
#SBATCH --account hai_fedak
#SBATCH --partition booster
#SBATCH --mail-user=tejumade.afonja@cispa.de
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mem 32G
#SBATCH --cpus-per-task=4
#SBATCH -o output1000_100000_TFs10_GT0-%x.txt 
#SBATCH -e error1000_100000_TFs10_GT0-%x.txt

nvidia-smi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /p/project/hai_fedak/teju/GRouNdGAN/activate.sh
cd /p/project/hai_fedak/teju/GRouNdGAN
# python src/main.py --config /p/project/hai_fedak/teju/GRouNdGAN/configs/causal_gan.cfg --preprocess --create_grn
# python src/main.py --config /p/project/hai_fedak/teju/GRouNdGAN/configs/causal_gan.cfg --create_grn
# tensorboard --logdir="/p/project/hai_fedak/teju/GRouNdGAN/results/GRouNdGAN/TensorBoard" --host 0.0.0.0 --load_fast false &
srun --exclusive python src/main.py --config /p/project/hai_fedak/teju/GRouNdGAN/configs/causal_gan.cfg --train&
wait
