#!/bin/bash -l

#SBATCH --job-name=nvidia
#SBATCH --partition=gpu
#SBATCH --mem=12GB
#SBATCH --output=output.txt
#SBATCH --gres=gpu:3
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=1-00:05:00

# Get conda
#module load miniconda
#module load anaconda3-2020.07-gcc-9.3.0-myrjwlf
#eval "$(conda shell.bash hook)"

echo "SLURM_JOBID="$SLURM_JOBID
echo " "
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo " "
# echo $PATH
# echo " "
# echo $PATH | grep test_build_2point2
# echo " "
# echo $(conda info --envs)

nvidia-smi

conda activate test_build_2point3

echo "==============================="
conda config --show channels
echo "==============================="

python test.py

