#!/bin/bash
#SBATCH -t 7-00:00:00
#SBATCH -A C3SE2019-1-14 # group to which you belong
#SBATCH -p vera  # partition (queue)
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=gpu:1
# modules already loaded
# jupyter notebook --no-browser  --ip=0.0.0.0 --port 8888
source $HOME/loadenv_gpu.sh
cd /c3se/users/zrimec/Vera/projects/DeepExpression/2019_2_22
snakemake -j 1 --latency-wait 22 --max-jobs-per-second 1 --forceall --resources gpu=200 mem_frac=160 > _run_hyperas_scerevisiae_l2.log  


