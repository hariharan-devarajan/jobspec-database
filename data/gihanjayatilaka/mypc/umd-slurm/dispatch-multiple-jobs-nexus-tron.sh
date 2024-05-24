#!/bin/bash
#SBATCH --job-name=pytorchjob
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32gb
#SBATCH --qos=high
#SBATCH --time=6:00:00
#SBATCH --output=/vulcanscratch/gihan/umd-slurm/logs/outFile-%A_%a.txt
#SBATCH --error=/vulcanscratch/gihan/umd-slurm/logs/errorFile-%A_%a.txt
#SBATCH --array=9-16

CONDA_ENV_NAME="longtails"
DIRECTORY="/vulcanscratch/gihan/long-tails/"
# DIRECTORY="/vulcanscratch/gihan/long-tails-relatedwork/scan2"


# CONDA_ENV_NAME="dac"
# DIRECTORY="/vulcanscratch/gihan/cmsc828project"


source ~/.bashrc
export TORCH_HOME=/vulcanscratch/gihan/torch-hub/
conda activate $CONDA_ENV_NAME
cd $DIRECTORY





sed -n "${SLURM_ARRAY_TASK_ID}p" < /vulcanscratch/gihan/umd-slurm/list-of-commands.sh
sed -n "${SLURM_ARRAY_TASK_ID}p" < /vulcanscratch/gihan/umd-slurm/list-of-commands.sh >&2
echo "------"
echo "------" >&2
# nvidia-smi
eval $(sed -n "${SLURM_ARRAY_TASK_ID}p" < /vulcanscratch/gihan/umd-slurm/list-of-commands.sh)


echo "END of SLURM commands"
echo "END of SLURM commands" >&2
