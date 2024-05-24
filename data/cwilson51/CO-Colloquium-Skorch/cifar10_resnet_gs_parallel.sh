#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=0-01:30
#SBATCH --output "parallel-multigpu-%j.out"


# set up environment
module load python
module list
source /path/to/your/env/bin/activate


# start dask scheduler and workers
echo 'Starting scheduler'
dask scheduler &
sleep 10

echo 'Scheduler booted, launching workers'
CUDA_VISIBLE_DEVICES=0 dask worker 127.0.0.1:8786 --nthreads 1 &
sleep 10
CUDA_VISIBLE_DEVICES=1 dask worker 127.0.0.1:8786 --nthreads 1 &

# run benchmarking script
python cifar10_resnet_parallel.py --max_epochs 50 --batch_size 2000
