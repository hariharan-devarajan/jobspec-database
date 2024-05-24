#!/bin/bash 
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --job-name=snakemake
cd /homes/bilala/BedTools/Chipseq_2024/scripts/snake_file_yam
snakemake --use-conda --cores 4 -s snake_file_with_yaml.smk --latency-wait 100
