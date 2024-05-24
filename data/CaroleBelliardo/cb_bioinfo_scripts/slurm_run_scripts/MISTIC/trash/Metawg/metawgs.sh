#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=60     # number of CPU per task #4
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=464G   # memory per Nodes   #38
#SBATCH -J "SR"   # job name
#SBATCH --mail-user=carole.belliardo@inrae.fr   # email address
#SBATCH --mail-type=ALL
#SBATCH -e slurm-SR-%j.err
#SBATCH -o slurm-SR-%j.out
#SBATCH -p all


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load singularity/3.5.3
module load nextflow/21.04.1
cd '/kwak/hub/25_cbelliardo/25_MISTIC/tools'


nextflow run -profile singularity metagwgs/main.nf \
--skip_host_filter  \
--type 'SR' \
--quality_type "illumina" \
--adapter1 "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA" \
--adapter2 "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT" \
--kaiju_db \
--diamond_bank "/database/hub/NR/NR_diamond/NR_2020_01_diamond.dmnd" \
--gtdbtk_bank "/database/hub/GTDB/release20211115/"  \
--eggnogmapper_db_download \
--input "/kwak/hub/25_cbelliardo/25_MISTIC/LRvsSR/list_wgstool.csv"  
