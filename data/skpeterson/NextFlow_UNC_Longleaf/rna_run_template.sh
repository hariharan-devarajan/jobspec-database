#!/bin/bash

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=20g
#SBATCH --time=2-00:00 # time (D-HH:MM)
#SBATCH --job-name=rna_nf
#SBATCH --mail-user=email@unc.edu
#SBATCH --mail-type=end
#SBATCH --output=/path/to/slurm_out/slurm_%j.out

## go to directory where you'd like temp files stored
cd /work/users/path/to/work

## Load Nextflow environment modules
module load nextflow/23.04.2;
module load apptainer/1.2.2-1;

nextflow run nf-core/rnaseq -r 3.14.0 -profile unc_longleaf -params-file /path/to/parameter_file.yaml

