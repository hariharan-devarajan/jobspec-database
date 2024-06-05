#!/bin/bash -l
#COBALT -t 60
#COBALT -q single-gpu
#COBALT -n 1
#COBALT --attrs filesystems=home,grand

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series
cd /home/amakhanov/ai-science-training-series/05_dataPipelines

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
python train_net_128_1.py
python train_net_128_2.py
python train_net_128_4.py
python train_net_128_8.py

python train_net_64_1.py
python train_net_64_2.py
python train_net_64_4.py
python train_net_64_8.py

python train_net_32_1.py
python train_net_32_2.py
python train_net_32_4.py
python train_net_32_8.py
