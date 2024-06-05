#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --account=cheme_gpu
#SBATCH --job-name=namd
#SBATCH --output=US-%j.out
#SBATCH --error=US-%j.error
#SBATCH --mem-per-cpu=2G

#SBATCH --priority=TOP

module purge;
ulimit -Sn 4096;


module load NAMD cuda

export PATH=/home/junwoony/Desktop/wham/wham:$PATH
export PATH=/home/zulissi/software/namd/Linux-x86_64-icc/:$PATH
namd2 +p4 +isomalloc_sync run_namd.conf
