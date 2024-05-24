#!/bin/bash
#SBATCH --partition=gpu    # Choose either "gpu" or "infer" node type
#SBATCH --nodes=1          # Resources from a single node
#SBATCH --gres=gpu:1       # One GPU per node (plus 25% of node CPU and RAM per GPU)
#SBATCH --time=00:01:00

#SBATCH --account=bdlds01

# ----------

# 1. activate the appropriate conda environment
export SLURM_EXPORT_ENV=ALL

# source activate tf_bede
source activate pytorch_bede

# 2. append the path for the conda environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# 3. check nvidia setup and gpu availability
nvidia-smi
# python test_if_gpu_available_tf.py
python test_if_gpu_available_pytorch.py

# ----------

# 4. run the python script
# srun python tensorflow_mnist_example.py --use-gpu True --num-workers 1

