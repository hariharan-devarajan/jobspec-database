#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=150
#SBATCH --mem=64G
#SBATCH --job-name=trust4
#SBATCH --time=336:00:00
#SBATCH --partition=general
#SBATCH --account=a_kelvin_tuong
#SBATCH -o trust4run_phs002517.output
#SBATCH -e trust4run_phs002517.error

output_dir="$1"
bam_files_dir="$2"
reference_file="$3"

srun trust4run.sh "$output_dir" "$bam_files_dir" "$reference_file"
