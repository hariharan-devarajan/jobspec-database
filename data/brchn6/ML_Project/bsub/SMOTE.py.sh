#!/bin/bash
#BSUB -q new-short               # Queue name
#BSUB -R "rusage[mem=16000]"     # Requested memory
#BSUB -J SMOTE_JOB           # Job name
#BSUB -o SMOTE_GPU_JOB_%J.out    # Output file
#BSUB -e SMOTE_GPU_JOB_%J.err    # Error file


# Load necessary modules
module load python  # Adjust based on your environment
module load miniconda

# Set up your environment
conda activate mlproject

# set directory to your working directory
cd /home/labs/cssagi/barc/FGS_ML/ML_Project

# Run your Python script
python /home/labs/cssagi/barc/FGS_ML/ML_Project/pyScripts/SMOTE.py

