#!/bin/bash -l

#SBATCH --job-name="cuFFS"
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --time=01:00:00
#SBATCH --clusters=garrawarla
#SBATCH --partition=gpuq
#SBATCH --account=mwaeor
#SBATCH --gres=gpu:1
#SBATCH --output=cuffs-%A.out
#SBATCH --error=cuffs-%A.err

source /pawsey/mwa/software/python3/build_base.sh
module load cuda
module load hdf5
module load cfitsio

module use /astro/mwaeor/achokshi/software/modulefiles
module load cuFFS

CUFFS_IN=$1

time rmsynthesis $CUFFS_IN
rm $CUFFS_IN
