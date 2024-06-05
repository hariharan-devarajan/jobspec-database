#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu
#SBATCH --account=cheme_gpu
#SBATCH --job-name=namd
#SBATCH --output=fw_vasp-%j.out
#SBATCH --error=fw_vasp-%j.error
#SBATCH --mem-per-cpu=2G

#SBATCH --priority=TOP

module purge; 
ulimit -Sn 4096; 

module load NAMD cuda

namd2 +p8 +isomalloc_sync fixed_mos2_solvate.namd
