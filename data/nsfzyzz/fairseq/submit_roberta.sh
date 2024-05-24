#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=40 # number of cores per task
#SBATCH --gres=gpu:4
##SBATCH --nodelist=bombe # if you need specific nodes
#SBATCH --exclude=ace,blaze,flaminio,freddie,r[10,16],atlas,havoc,steropes
#SBATCH -t 3-00:00 # time requested (D-HH:MM)
#SBATCH -D /data/yyaoqing/fairseq/
#SBATCH -o /home/eecs/yyaoqing/slurm_logs/slurm.%N.%j..out # STDOUT
#SBATCH -e /home/eecs/yyaoqing/slurm_logs/slurm.%N.%j..err # STDERR
pwd
hostname
date
echo starting job...
source ~/.bashrc
conda activate pytorch-transformer
export PYTHONUNBUFFERED=1

size_ratio=0.5
CKPT_DIR=/data/yyaoqing/fairseq/checkpoint/roberta_pretrain_wikitext-103_subsample_$size_ratio/
DATA_DIR=/data/yyaoqing/fairseq/data-bin/wikitext-103
LOG_DIR=/home/eecs/yyaoqing/logs/roberta_logs/

srun -N 1 -n 1 --gres=gpu:4 fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name base task.data=$DATA_DIR \
1>$LOG_DIR/train_$size_ratio.log 2>$LOG_DIR/train_$size_ratio.err &

wait
date
