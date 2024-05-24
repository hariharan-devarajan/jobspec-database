#!/bin/bash


#SBATCH --qos=6hours
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test_AMRtcp


ml load Java/13.0.2
# source /scicore/home/egliadr/leeman0000/.bashrc
# conda activate nextflow

NF_DIR="/scicore/home/egliadr/leeman0000/tools"
MAIN_DIR="/scicore/home/egliadr/leeman0000/github/AMRtcp"

$NF_DIR/nextflow run $MAIN_DIR/main.nf -with-singularity -with-report -profile slurm
