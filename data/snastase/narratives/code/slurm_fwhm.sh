#!/bin/bash

# Run within BIDS code/ directory:
# sbatch slurm_fwhm.sh

# Set partition
#SBATCH --partition=all

# How long is job (in minutes)?
#SBATCH --time=240

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=1 --mem-per-cpu=12000

# Name of jobs?
#SBATCH --job-name=3dFWHMx

# Where to output log files?
#SBATCH --output='../derivatives/logs/3dFWHMx-%A_%a.log'

# Number jobs to run in parallel, pass index as subject ID
#SBATCH --array=1-345

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Load AFNI module
echo "Loading AFNI module afni/2019.10.09"
module load afni/2019.10.09
afni --version

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date

# Set subject ID based on array index
printf -v subj "%03d" $SLURM_ARRAY_TASK_ID

# Run fMRIPrep script with participant argument
echo "Running spatial smoothing estimate for sub-$subj"

./get_fwhm.py $subj

echo "Finished smoothing estimate for sub-$subj"
date