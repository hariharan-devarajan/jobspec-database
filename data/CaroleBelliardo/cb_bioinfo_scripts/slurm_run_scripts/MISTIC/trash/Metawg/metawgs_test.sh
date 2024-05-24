#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=60     # number of CPU per task #4
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=464G   # memory per Nodes   #38
#SBATCH -J "Run1"   # job name
#SBATCH --mail-user=carole.belliardo@inrae.fr   # email address
#SBATCH --mail-type=ALL
#SBATCH -e slurm-run1-%j.err
#SBATCH -o slurm-run1-%j.out
#SBATCH -p all


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load singularity/3.7.3
module load nextflow/21.04.1

cd /kwak/hub/25_cbelliardo/MISTIC/
nextflow run -profile test,genotoul metagwgs/main.nf \
--type 'SR' \
--input 'metagwgs-test-datasets/small/input/samplesheet.csv' \
--skip_host_filter --skip_kaiju --stop_at_clean
