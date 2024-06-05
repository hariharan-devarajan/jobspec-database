#!/bin/bash

#SBATCH --job-name=gt3x_processing
#SBATCH --array=1-100%50
#SBATCH --output=/ocean/projects/med220004p/shared/data_sandbox/ggir_proc/group1/logs/job_%A_%a.out
#SBATCH --error=/ocean/projects/med220004p/shared/data_sandbox/ggir_proc/group1/logs/job_%A_%a.err
#SBATCH --nodes 1
#SBATCH --partition RM-shared
#SBATCH --time 10:00:00
#SBATCH --ntasks-per-node 10
set -x
# Your singularity container path
SINGULARITY_CONTAINER=/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/ggir_test_ggir_test2_v2.sif 

# Base directory for the sub-directories
BASE_DIR=/ocean/projects/med220004p/shared/data_sandbox/ggir_proc/group1/

# Calculate the sub-directory to process based on the SLURM_ARRAY_TASK_ID
DIR_TO_PROCESS="${BASE_DIR}/subdir_${SLURM_ARRAY_TASK_ID}"

# Run the singularity container on the sub-directory
# Bind the DIR_TO_PROCESS to both /data and /output in the container

#singularity run --bind /ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/external_ID/tmp:/data,/ocean/projects/med220004p/shared/data_raw/backup_onprem/adam/external_ID/tmp:/output 
singularity run --bind ${DIR_TO_PROCESS}:/data --bind ${DIR_TO_PROCESS}:/output ${SINGULARITY_CONTAINER} 