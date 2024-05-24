#!/bin/bash
# Remember to change this to your own project ID after the course!
#SBATCH -A hpc2n2024-025
# We are asking for 5 minutes
#SBATCH --time=00:05:00
# Asking for one GPU
#SBATCH --gres=gpu:v100:1

# Remove any loaded modules and load the ones we need
module purge  > /dev/null 2>&1
module load GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1

# DO THIS BEFORE SUBMITTING!
# BEGIN
# Create a virtual environment and install sklearn
# Do the following on Kebnekaise:
# ml GCC/10.3.0  OpenMPI/4.1.1 TensorFlow/2.6.0-CUDA-11.3.1
# python -m venv --system-site-packages scikit-venv
# source scikit-venv/bin/activate
# pip install --no-cache-dir --no-build-isolation scikit-learn
# deactivate
# Now you can submit with "sbatch scikit-learn-kebnekaise.sh" 
# END 

# Activate the virtual environment we installed to
source scikit-venv/bin/activate

# Run your Python script
python scikit-learn-kebnekaise.py
