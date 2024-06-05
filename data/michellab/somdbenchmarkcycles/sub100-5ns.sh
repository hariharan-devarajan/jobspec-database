#!/bin/bash
#SBATCH -o somd-array-gpu-%A.%a.out
#SBATCH -p main
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00

echo "CUDA DEVICES:" $CUDA_VISIBLE_DEVICES

mkdir 100cycles-5ns
cd 100cycles-5ns

export OPENMM_PLUGIN_DIR=/home/julien/sire.app/lib/plugins/

srun ~/sire.app/bin/somd-freenrg -C ../sim100-5ns.cfg -l 0.50 -p CUDA
cd ..


