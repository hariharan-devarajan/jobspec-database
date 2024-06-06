#!/bin/bash
#SBATCH -t 04:00:00
#SBATCH --mem=16GB
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -J xnat2bids
#SBATCH -o /gpfs/scratch/%u/logs/xnat2bids-%A_%a.out
#SBATCH --array=148

#--------- Variables ---------
# This line makes our bash script complaint if we have undefined variables
set -u

# Read variables in the .env file in current directory
# This will read:
# XNAT_USER, XNAT_PASSWORD
set -a
[ -f .env ] && . .env
set +a
#--------- xnat-tools ---------

# version of xnat2bids being used
version=v1.0.5
# Path to Singularity Image for xnat-tools (maintained by bnc)
simg=/gpfs/data/bnc/simgs/brownbnc/xnat-tools-${version}.sif

#--------- directories ---------

data_dir=/gpfs/data/bnc

# Output directory
output_dir=${data_dir}/shared/bids-export/${USER}

mkdir -m 775 ${output_dir} || echo "Output directory already exists"

# The bidsmap file for your study lives under ${bidsmap_dir}/${bidsmap_file}
bidsmap_dir=/users/$USER/src/bnc/shenhav_201226/preprocessing/xnat2bids
bidsmap_file=shenhav_201226.json
tmp_dir=/gpfs/scratch/$USER/xnat2bids

mkdir -m 775 ${tmp_dir} || echo "Temp directory already exists"

#----------- Dictionaries for subject specific variables -----
# Dictionary of sessions to subject
declare -A sessions=([200]="XNAT7_E00063" \
                     [148]="XNAT_E00005" )

# Dictionary of series to skip per subject
# Series 6 corresponds to multi-echo mprage. 
# fmriprep only need the RMS (series 7). 
# Skipping series 6 keeps BIDS-validation happy
declare -A skip_map=([200]="-s 6"\
                     [148]="-s 6")

# Use the task array ID to get the right value for this job
XNAT_SESSION=${sessions[${SLURM_ARRAY_TASK_ID}]}
SKIP_STRING=${skip_map[${SLURM_ARRAY_TASK_ID}]}

echo "Processing session:"
echo ${XNAT_SESSION}
echo "Series to skip:"
echo ${SKIP_STRING}

#--------- Run xnat2bids ---------
# runs singularity command to extract DICOMs from xnat and export to BIDS
# this command tells singularity to launch out xnat-tools-${version}.sif image
# and execute the xnat2bids command with the given inputs.
# The `-B` flag, binds a path. i.e, makes that directory available to the singularity container
# The file system inside your container is not the same as in Oscar, unless you bind the paths
# The -i passes a sequence to download, without any -i all sequences will be processed
singularity exec --no-home -B ${output_dir} -B ${bidsmap_dir}:/bidsmaps:ro ${simg} \
    xnat2bids ${XNAT_SESSION} ${output_dir} \
    -u ${XNAT_USER} \
    -p "${XNAT_PASSWORD}" \
    --overwrite \
    ${SKIP_STRING} 
    
