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

cd '/kwak/hub/25_cbelliardo/MISTIC/Salade_I/2_QC_fastq/'

nextflow run -profile genotoul metagwgs/main.nf \ #--skip_host_filter  \
--skip_host_filter \
--skip_kaiju \
--type 'SR' \
--quality_type "illumina" \
--adapter1 "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA" \
--adapter2 "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT" \ #--kaiju_db \
--stop_at_clean \ #--diamond_bank "/database/hub/NR/NR_diamond/NR_2020_01_diamond.dmnd" \ #--gtdbtk_bank "/database/hub/GTDB/release20211115/"  \ #--eggnogmapper_db_download \
--input "/kwak/hub/25_cbelliardo/MISTIC/Salade_I/2_QC_fastq/list_wgstool.csv"  
