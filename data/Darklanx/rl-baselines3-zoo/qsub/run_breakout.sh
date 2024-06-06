#!/bin/bash 
#PBS -l select=1:ncpus=40
#PBS -N run_breakout
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

python /home/nycucpu1/minatar/rl-baselines3-zoo/train.py --algo offpac --env breakout -min -n 3000000 -params KL:True  --seed $seed
