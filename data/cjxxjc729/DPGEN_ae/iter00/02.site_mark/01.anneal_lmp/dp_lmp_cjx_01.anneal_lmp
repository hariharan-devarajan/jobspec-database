#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --account=slchen
#SBATCH --gres=gpu:1
export OMP_NUM_THREADS=1
module load nvidia/cuda/10.1
module load intel/parallelstudio/2017u8
/project/chenyongtin/tools/anaconda3/envs/deepmd-kit/bin/lmp -i in.lammps >lmp.out

