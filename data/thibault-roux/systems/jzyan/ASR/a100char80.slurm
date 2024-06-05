#!/bin/bash
#SBATCH --job-name=char80
#SBATCH --output=output/char80.txt
#SBATCH --mail-user=thibault.baneras.roux@gmail.com
#SBATCH --mail-type=END
#SBATCH --time=98:35:00
#SBATCH -C a100
#SBATCH -A rbg@a100
#SBATCH --gpus-per-node=4

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu-t4

# avec qps_gpu-t4, on peut mettre à 99h la limite de temps


# python -m torch.distributed.launch --nproc_per_node=2 train.py hparams/char80.yaml --distributed_launch --distributed_backend='nccl'

python train.py hparams/char80.yaml --data_parallel_backend

