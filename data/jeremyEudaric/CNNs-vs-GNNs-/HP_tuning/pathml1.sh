#!/bin/bash



#SBATCH -J split_1


#SBATCH --output split_1-%J.out

#SBATCH --error split_1-%J.err


#SBATCH --time=100:00:00

#SBATCH --gres gpu:1
#SBATCH --mem=12G
##SBATCH --no-requeue
#SBATCH -p rocmnodes

spack load singularity@3.8.5

singularity exec --bind  /mnt/scratchc/fmlab/eudari01/BREAST_METABRIC2:/mnt -B /mnt/scratchc/fmlab/eudari01/trial:/data  packages.img4 python /mnt/Vgg19_Regression_each_fold_2.py

