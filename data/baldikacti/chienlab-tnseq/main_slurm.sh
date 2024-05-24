#!/usr/bin/bash
#SBATCH --job-name=chienlab-tnseq-ba   # Job name
#SBATCH --partition=cpu            # Partition (queue) name
#SBATCH --ntasks=24                   # Number of CPU cores
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --mem=240gb                     # Job memory request
#SBATCH --time=06:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/chienlab-tnseq-ba_%j.log   # Standard output and error log
date;hostname;pwd

# Load modules

module load miniconda/22.11.1-1

# Activate conda environment

conda activate /work/pi_pchien_umass_edu/berent/chienlab-tnseq/conda-tnseq

# Run pipeline with all available cores
# Empty --cores argument defaults to all available cores
snakemake -q rules --profile profiles/default

date