#!/bin/bash
#SBATCH --job-name=IMG-DDPM
#SBATCH --time=30-00:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --mem=50G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=logs/%j.out
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export load_path=data/imgs/train
export dim=64
export image_size=128
export timesteps=1000
export sampling_timesteps=1000
export objective=pred_noise
export beta_schedule=sigmoid
export channels=3
export resnet_block_groups=8
export learned_sinusoidal_dim=16
export attn_dim_head=32
export attn_heads=4
export batch_size=32
export lr=8e-5
export num_steps=700000
export gradient_accumulate_every=2
export ema_decay=0.995
export save_and_sample_every=1000


python \
    train.py \
    --dim $dim \
    --image_size $image_size \
    --timesteps $timesteps \
    --sampling_timesteps $sampling_timesteps \
    --objective $objective \
    --beta_schedule $beta_schedule \
    --channels $channels \
    --resnet_block_groups $resnet_block_groups \
    --learned_sinusoidal_dim $learned_sinusoidal_dim \
    --attn_dim_head $attn_dim_head \
    --attn_heads $attn_heads \
    --load_path $load_path \
    --batch_size $batch_size \
    --lr $lr \
    --num_steps $num_steps \
    --gradient_accumulate_every $gradient_accumulate_every \
    --ema_decay $ema_decay \
    --save_and_sample_every $save_and_sample_every \
    --wandb