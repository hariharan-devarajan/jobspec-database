#!/bin/bash --login

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --job-name=trust4_fastq
#SBATCH --time=96:00:00
#SBATCH --partition=general
#SBATCH --account=a_kelvin_tuong
#SBATCH -o trust4run_fastq_phs002599.output
#SBATCH -e trust4run_fastq_phs002599.error

# Set variables
dir_path="$2"
output_dir="$1"
reference_file="$3"

srun trust4_run_fastq.sh "$output_dir" "$dir_path" "$reference_file"
