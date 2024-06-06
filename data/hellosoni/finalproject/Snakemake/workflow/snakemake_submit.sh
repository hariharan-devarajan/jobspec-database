#!/bin/bash

#SBATCH --account=PAS1855
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1

set -e -u -o pipefail

module load python/3.6-conda5.2
source activate ipy-env

snakemake -j"$SLURM_NTASKS" -p  