#!/bin/bash
#SBATCH -A hpc2n2024-025 # Change to your own after the course
#SBATCH --time=00:10:00 # Asking for 10 minutes
# Asking for one V100 card
#SBATCH --gres=gpu:v100:1

# Load any modules you need
module load GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 SciPy-bundle/2021.10 TensorFlow/2.7.1

# If you are using the example named "my_tf_program.py" in this directory, 
# you need to create and activate a virtual environment containing 
# scikit learn. Do these steps: 
#
# module load GCC/11.2.0 OpenMPI/4.1.1 Python/3.9.6 SciPy-bundle/2021.10 TensorFlow/2.7.1
# virtualenv --system-site-packages <path-to-install-dir>/tf-venv
# source <path-to-install-dir>/tf-venv/bin/activate
# pip install --no-cache-dir --no-build-isolation scikit-learn 
#
# Then remove the below outcommenting so the virtual environment gets activated in the batch job (and add the correct path) 
# source <path-to-install-dir>/tf-venv/bin/activate

# Run your Python script
python my_tf_program.py
