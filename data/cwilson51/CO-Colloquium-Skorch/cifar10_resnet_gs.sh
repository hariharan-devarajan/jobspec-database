#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --gpus-per-node=1
#SBATCH --mem=5000M
#SBATCH --time=0-03:00

# set up environment
module load python
module list
source /path/to/your/env/bin/activate

# run benchmarking script
python cifar10_resnet.py --max_epochs 50 --n_jobs 1 --batch_size 2000
