#!/bin/sh
#PBS -q h-short
#PBS -Wgroup_list=gk36
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1
#PBS -l walltime=02:00:00
#PBS -N run
####################

cd $PBS_O_WORKDIR
. env.sh

# 実行したいコマンド
python example.py
