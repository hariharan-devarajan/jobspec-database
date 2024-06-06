#!/bin/sh
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --container-name=nvidia_pytorch_23_06
#SBATCH --container-writable
#SBATCH --container-remap-root
#SBATCH --container-mounts=/etc/slurm/task_prolog:/etc/slurm/task_prolog,/scratch:/scratch,/pfs/work7/workspace/scratch/hd_oy280-pipe:/workspace
#SBATCH --error=error_job
#SBATCH --output=output_job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nilsmailiseke@gmail.com
#SBATCH --job-name=nnUnet_job

# apt-get install graphviz
# /usr/bin/python -m pip install --upgrade pip
# . install_nnUnet.sh
# . install_hiddenlayers.sh
. setup_paths.sh
. preprocess.sh
# echo starting training
. resume_training.sh
. start_multi_gpu_training_2d.sh
