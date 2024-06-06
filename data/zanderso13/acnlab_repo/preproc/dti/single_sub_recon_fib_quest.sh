#!/bin/bash

#SBATCH -A p30954
#SBATCH -p short
#SBATCH -t 4:00:00
#SBATCH --mem=32G
#SBATCH -J QSDR_reconstruction_single_sub

# Reconstruction Parameters                                                                                           
param0="1.25"
record_odf="1"
reg_method="1"
scheme_balance="1"
check_btable="1"
voxel_res="2"     
thread="4"

module load singularity/latest

singularity exec /home/zaz3744/ACNlab/software/singularity_images/dsi-studio-docker.sif /dsistudio/dsi_studio_64/dsi_studio --action=rec --thread=${thread} --source=${1} --method=7 --param0=${param0} --odf_order=8 --param1=${voxel_res} --motion_correction=1 --output_jac=1 --output_map=1 --record_odf=1 --reg_method=${reg_method}
