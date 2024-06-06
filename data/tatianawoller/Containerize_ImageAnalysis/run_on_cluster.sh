#!/bin/bash
#

# Lines starting with # are comments, and will not be run.
# Lines starting with #SBATCH specify options for the scheduler.
# Lines that do not start with # or #SBATCH are commands that will run.

# Name for the job that will be visible in the job queue and accounting tools.
#SBATCH --job-name runctonaineraigj

# Name of the SLURM partition that this job should run on.
#SBATCH -p 32GB       # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 2

# Memory (RAM) requirement/limit in MB.
#SBATCH --mem 28672      # Memory Requirement (MB)

# Time limit for the job in the format Days-H:M:S
# A job that reaches its time limit will be cancelled.
# Specify an accurate time limit for efficient scheduling so your job runs promptly.
#SBATCH -t 0-4:0:00

# The standard output and errors from commands will be written to these files.
# %j in the filename will be replace with the job number when it is submitted.
#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

# Send an email when the job status changes, to the specfied address.
#SBATCH --mail-type ALL
#SBATCH --mail-user [your email address]

#load Singularity
module load singularity/3.5.3

# COMMAND GROUP 1 - go to the folder where the container is saved
cd [your folder with the container]

#define here parameters for your tool 
imag_directory="[file path to images]"
imag_savesegmented="[file path to save segmentations]"
mode="nuclei"
flow_threshold=0
cellprob_threshold=-1
celldiameter=19
channel=1

# COMMAND GROUP 2 - run the container with your settings
singularity run cellpose_container.sif --filedir $imag_directory --savedir $imag_savesegmented --pretrained_model $mode --flow_threshold $flow_threshold --cellprob_threshold $cellprob_threshold --diameter $celldiameter --chan $channel --save_tif




# END OF SCRIPT
