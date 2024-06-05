#!/bin/bash
#SBATCH -N 1
#SBATCH -J launch       # Job Name
#SBATCH -o setup_sub_fs_137 # Name of the output file (eg. myMPI.oJobID)
#SBATCH -p normal
#SBATCH -t 10:00:00
#SBATCH -n 2
#SBATCH --get-user-env
#SBATCH -A Analysis_Lonestar
#----------------
# Job Submission
#----------------
umask 2

module load launcher
module load fsl
module load freesurfer
module load matlab
## DO NOT EDIT
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_WORKDIR=.
## OK YOU CAN EDIT NOW

## THIS MUST BE A COMPLETE PATH TO THE JOB FILE
export LAUNCHER_JOB_FILE=$WORK/setup_subject/run_setupfs.sh

$LAUNCHER_DIR/paramrun
