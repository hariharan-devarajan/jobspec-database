#!/bin/bash

#SBATCH -A p30954
#SBATCH -p short
#SBATCH -t 00:10:00
#SBATCH --mem=32G
#SBATCH -J dti_create_src

# You're about to run a singularity image... so you need singularity
module load singularity/latest

# USER SPECIFICATION
# action is the first input to dsi studio
action=src

# source is going to refer dynamically to each BIDS formatted directory where the nii file is located
# note: This will be the full path to the file name and the name itself
source=$1

# nii files need to have bval and bvec files designated separately because this information is not included in the header of the nii file itself. 
# Each variable should be equal to to the file type or basically what comes after the . in the file name
bval=$3
bvec=$2

# the recursive option will be helpful for future projects. It will grab all files in subdirectories. 
recursive=0

echo "Work on the following folder: $1"

# DSI studio when working from a singularity image struggles with paths. It's important that the working directory (where you open dsi studio from) is a high enough directory such that all data is housed in sub-directories
projdir=/projects/b1108

cd ${projdir}

singularity exec /home/zaz3744/ACNlab/software/singularity_images/dsi-studio-docker.sif /dsistudio/dsi_studio_64/dsi_studio --action=$action --source=$source --bval=${bval} --bvec=${bvec}
