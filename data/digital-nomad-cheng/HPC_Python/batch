#!/usr/bin/env bash
##SBATCH -A C3SE2022-1-9
#SBATCH -A C3SE2023-1-8
##SBATCH --gres=gpu:T4:1 
#SBATCH --gres=gpu:V100:1
##SBATCH --gres=gpu:A100:1
#SBATCH -t 1-23:00:00
##SBATCH -t 0-00:30:00
#SBATCH -n 1
#SBATCH -c 32
module purge
ml AMGX/2.3.0-foss-2021a-CUDA-11.3.1 SciPy-bundle/2021.05-foss-2021a matplotlib/3.4.2-foss-2021a
ml PyTorch/1.12.1-foss-2021a-CUDA-11.3.1
ml scikit-learn/0.24.2-foss-2021a
./run-python

