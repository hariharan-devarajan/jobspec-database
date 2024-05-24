#!/bin/bash
#SBATCH -J snakemake
#SBATCH -o out_snakemake
#SBATCH -e err_snakemake
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time 0-01:00:00        # DAYS-HOURS:MINUTES:SECONDS

source ~/miniforge3/etc/profile.d/conda.sh
# activate any conda environment that has snakemake installed
conda activate bioinformatics

# Store the location of where you've installed this github repository
GIT_REPO_DIR=/scratch/gpfs/bjarnold/intro_compbio_workflows_2024

snakemake --directory ../ --snakefile ../Snakefile \
--profile ${GIT_REPO_DIR}/snakemake_profiles/slurm

