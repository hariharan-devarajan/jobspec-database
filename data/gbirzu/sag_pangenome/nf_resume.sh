#!/bin/bash
#SBATCH --job-name=resume_nf
#SBATCH --mail-type=END         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=gbirzu@stanford.edu # Where to send mail
#SBATCH --time=24:00:00 # Time limit hrs:min:sec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

WORKFLOW=$1
PROFILE=$2

ml ncbi-blast+

nextflow run ${WORKFLOW} -profile ${PROFILE} -resume
