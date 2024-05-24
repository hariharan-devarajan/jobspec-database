#!/bin/bash
#SBATCH --job-name=pytorch_mnist     # job name
#SBATCH --ntasks=4                   # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=3:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=pytorch_mnist%j.out # output file name
#SBATCH --error=pytorch_mnist%j.err  # error file name
#SBATCH --array=4,6,8 
set -x
cd ${SLURM_SUBMIT_DIR}
export WANDB_MODE="offline"
module purge

module load anaconda-py3/2021.05
conda activate /gpfswork/rech/mdb/urz96ze/miniconda3/envs/Barth
module load pytorch-gpu/py3/1.11.0
###module load pytorch-gpu/py3/1.4.0 
### for i in 4 8 12 16; do
###     python ./mnist_example.py --batch-size $i &

python ./mnist_example.py --batch-size ${SLURM_ARRAY_TASK_ID}