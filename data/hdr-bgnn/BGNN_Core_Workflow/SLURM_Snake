#!/bin/bash
#SBATCH --account=PAS2136
#SBATCH --job-name=Download_2_Morph
#SBATCH --time=02:30:00
#
# Runs Snakemake pipeline that downloads, crops, and segments images.
# There are three required positional arguments.
# Usage:
# sbatch SLURM_Snake <WORKDIR> <INPUTCSV> <>
# - SNAKEFILE - snakefile you want to used
# - WORKDIR - Snakemake working directory - contains output files
# - INPUTCSV - Path to input CSV file specifying images to process

# Stop if a command fails (non-zero exit status)
set -e

# Verify command line arguments
WORKDIR=$1
INPUTCSV=$2
NUMJOBS=$3

# Default to running 4 jobs if not set
if [ -z "$NUMJOBS" ]
then
      NUMJOBS=4
fi

# Ensures CSV path is an absolute path.
# Otherwise snakemake will look for this file within $WORKDIR.
REAL_INPUTCSV=$(realpath $INPUTCSV)

# Make sure the input CSV file exists
if [ ! -f "$REAL_INPUTCSV" ]
then
   echo "ERROR: Required INPUTCSV file $INPUTCSV does not exist."
   exit 1
fi

# Activate Snakemake environment
module load miniconda3/4.10.3-py37
# Activate using source per OSC instructions
source activate snakemake

# Setup Snakemake/sbatch to use same account as this job
export SBATCH_ACCOUNT=$SLURM_JOB_ACCOUNT

# Run pipeline using Snakemake
snakemake \
    --jobs $NUMJOBS \
    --profile slurm/ \
    --use-singularity \
    --directory $WORKDIR \
    --config list=$REAL_INPUTCSV
chmod -R 774 $WORKDIR
