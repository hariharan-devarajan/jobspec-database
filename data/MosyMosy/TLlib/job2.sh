#!/bin/bash
#SBATCH --mail-user=SLRUMReport@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=nprior_20_101
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-08:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/envs/TLlib/bin/activate

echo "change TORCH_HOME environment variable"
cd $SLURM_TMPDIR
cp -r ~/scratch/Pytorch_zoo .
export TORCH_HOME=$SLURM_TMPDIR/Pytorch_zoo

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/TLlib .

echo "Copying the datasets"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/TLlib_Dataset .

date +"%T"
echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cd TLlib
cd examples/domain_adaptation/image_classification

CUDA_VISIBLE_DEVICES=0 python nprior.py $SLURM_TMPDIR/TLlib_Dataset/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --seed 1 --log $SLURM_TMPDIR/TLlib/logs/nprior_20_101/Office31_A2W --ce_weight 0

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cp -r $SLURM_TMPDIR/TLlib/logs/nprior_20_101 ~/scratch/TLlib/logs/nprior_20_101