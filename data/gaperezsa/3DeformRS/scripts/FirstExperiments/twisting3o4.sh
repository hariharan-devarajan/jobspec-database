#!/bin/sh
#SBATCH --job-name=twist3Of4
#SBATCH --time=0-4:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/santamgp/Documents/CertifyingAffineTransformationsOnPointClouds/3D-RS-PointCloudCertifying/
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=13-18
#SBATCH --constraint=v100

module load gcc

echo "######################### SLURM JOB ########################"
echo HOST NAME
echo `hostname`
echo "############################################################"


environment=CertifyingPointClouds
conda_root=$HOME/anaconda3

source $conda_root/etc/profile.d/conda.sh
conda activate $environment

set -ex

# ------------------------ need not change -----------------------------------

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID))"p scripts/AllTwisting.txt)
python3 Certify.py  $LINE 