#!/bin/bash
#SBATCH --job-name=gvtp
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mem=10G
#SBATCH --mail-type=ALL

module load nextflow

SINGULARITY_TMPDIR=$PWD/tmp
export SINGULARITY_TMPDIR

TMPDIR=$PWD/tmpdir
export TMPDIR

nextflow run -hub gitlab PlantGenomicsLab/easel-benchmarking-v2-nf -profile singularity,xanadu -params-file gvtp.yaml


