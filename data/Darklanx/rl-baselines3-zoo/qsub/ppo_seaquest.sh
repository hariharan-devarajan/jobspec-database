#!/bin/bash 
#PBS -l select=1:ncpus=40
#PBS -N ppo_seaquest
#PBS -q ct160
#PBS -P MST110476 
#PBS -j eo
#PBS -l walltime=15:00:00
#PBS -o myscript.out

cd /home/nycucpu1/minatar/rl-baselines3-zoo
module load anaconda3/5.1.10
module load gcc/9.3.0 
module load mpi/openmpi-4.0.5/gcc930
module load cuda/10.0.130  
source activate /home/nycucpu1/.conda/envs/downgrade

python /home/nycucpu1/minatar/rl-baselines3-zoo/train.py --algo ppo --env seaquest -min -n 3000000 --seed $seed
# python train.py --algo ppo --env asterix -min -n 3000000 --seed $seed
