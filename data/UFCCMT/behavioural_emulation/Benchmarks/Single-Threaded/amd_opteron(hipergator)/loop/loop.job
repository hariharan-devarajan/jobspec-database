#!/bin/sh
#
#PBS -r n
#PBS -N loop
#PBS -o loop_log.out
#PBS -m e
#PBS -M kevincheng88@gmail.com
#PBS -l nodes=1:ppn=1
#PBS -l pmem=800mb
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR

cat /proc/cpuinfo

./run_loop.sh