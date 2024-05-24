#!/bin/bash

#SBATCH --job-name=bismark-mbd
#SBATCH --output=/home/lspencer/DuMOAR/sbatch/bismark-20230422.txt
#SBATCH --mail-user=laura.spencer@noaa.gov
#SBATCH --mail-type=ALL
#SBATCH -c 20
#SBATCH -t 21-0:0:0

# DuMOAR Project
# This script is for aligning trimmed MBD-BS data to Dungeness crab genome via bismark
#     and making methylation calls.
# Data was generated from Dungeness crab exposed to two pH conditions during early life.

module load bio/bismark/0.24.0
source /home/lspencer/venv/bin/activate

# path variables not working, trying to just provide hard paths
IN="/scratch/lspencer/DuMOAR/concatenated/after-trim/"
OUT="/scratch/lspencer/DuMOAR/aligned/trim-before-concat/"

# # Prepare genome for Bismark
# # NOTE: for some reason using a path variable to specify genome folder didn't work
# # Also- this only needs to be run once
bismark_genome_preparation \
--verbose \
--parallel 20 \
--bowtie2 \
/home/lspencer/references/dungeness/

# Run Bismark to align trimmed reads

# Move to where bismark files will be written
cd ${OUT}

# NOTE: bismark guide recommends only 4 cores

find ${IN}*_R1.fastq \
| xargs basename -s _R1.fastq | xargs -I{} bismark \
--bowtie2 \
-genome /home/lspencer/references/dungeness/ \
-p 4 \
-score_min L,0,-0.6 \
--non_directional \
-1 ${IN}{}_R1.fastq \
-2 ${IN}{}_R2.fastq \
--output_dir ${OUT}

# Deduplicate, call methylation status, extract methylation and create downstream amendable files.

find *.bam | \
xargs basename -s .bam | \
xargs -I{} deduplicate_bismark \
--bam \
--paired \
{}.bam

bismark_methylation_extractor \
--bedGraph --counts --scaffolds \
--multicore 20 \
--buffer_size 75% \
*deduplicated.bam

# Bismark processing report
bismark2report

#Bismark summary report
bismark2summary

# Run multiqc to summarize fastqc reports
multiqc . .

# NOTE- THE RESULTING BAM FILES ARE NOT SORTED!  NEED TO SORT THEM FOR METHYLKIT.
