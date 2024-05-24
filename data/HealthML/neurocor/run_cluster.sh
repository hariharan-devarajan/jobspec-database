#!/bin/bash

#SBATCH --job-name=controljob_%j
#SBATCH --output=snakemake_%j.log
#SBATCH --partition=vcpu,hpcpu
#SBATCH --time=24:00:00
#SBATCH -c 1
#SBATCH --mem 2000

SNAKEMAKE_ENV=snakemake

# Initialize conda:
eval "$(conda shell.bash hook)"
conda activate ${SNAKEMAKE_ENV}

snakemake --snakefile workflow/Snakefile \
          --configfile config/config.yaml \
	  --profile ./slurm \
          --rerun-triggers mtime \
          --directory "${PWD}" \
	  "${@}"


