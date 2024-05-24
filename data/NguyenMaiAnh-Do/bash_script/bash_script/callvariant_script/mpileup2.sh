#!/bin/bash
#
#SBATCH --job-name=mpileup_call_2
#SBATCH --output=/home/ndo/slurm_script/mpileup_%j.out
#SBATCH --error=/home/ndo/slurm_script/mpileup_%j.err

#

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem-per-cpu=4G
eval "$(conda shell.bash hook)"
conda activate nextflow

bed=/home/ndo/GIAB-GT/HG001_GRCh38_1_22_v4.2.1_benchmark.bed
ref=/home/ndo/GIABv3Ref/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta
prior=0.99
bam=/home/ndo/03_final_bam/SRR13586007_sortCoord.bam

#time bcftools mpileup -B -Q5 --max-BQ 30 -I -Ou --threads 28 -T $bed -f $ref $cram | bcftools call -P $prior --threads 28 -mv -Oz -o ~/PAO89685.pass.cram_highconf.vcf


bcftools mpileup -Q1 --max-BQ 60 -I -a FORMAT/AD -Ou --threads 28 -T $bed -f $ref $bam \
| bcftools call --threads 10 -Oz -mv -P $prior -o /home/ndo/04_callvariant_vcf/SRR13586007_bcftools.vcf