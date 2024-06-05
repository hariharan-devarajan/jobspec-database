#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH -o /data/biohub/20190129_Lardelli_FMR1_RNASeq/slurm/%x_%j.out
#SBATCH -e /data/biohub/20190129_Lardelli_FMR1_RNASeq/slurm/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stephen.pederson@adelaide.edu.au

## Cores
CORES=16

## Modules
module load FastQC/0.11.7
module load STAR/2.5.3a-foss-2016b
module load SAMtools/1.3.1-GCC-5.3.0-binutils-2.25
module load AdapterRemoval/2.2.1-foss-2016b
module load GCC/5.4.0-2.26

## Genomic Data Files
REFS=/data/biorefs/reference_genomes/ensembl-release-94/danio-rerio/
GTF=${REFS}/Danio-rerio.GRCz11.94.chr.gtf

## Directories
PROJROOT=/data/biohub/20190129_Lardelli_FMR1_RNASeq

## Setup for genome alignment
TRIMDATA=${PROJROOT}/1_trimmedData
ALIGNDATA=${PROJROOT}/2_alignedData
mkdir -p ${ALIGNDATA}/logs
mkdir -p ${ALIGNDATA}/bams
mkdir -p ${ALIGNDATA}/FastQC
mkdir -p ${ALIGNDATA}/featureCounts

## Aligning, filtering and sorting
for R1 in ${TRIMDATA}/fastq/*1.fq.gz
  do
  
  BNAME=$(basename ${R1%1.fq.gz})
  R2=${R1%1.fq.gz}2.fq.gz
  echo -e "STAR will align:\n\t${R1} amd \n\t${R2}"

    STAR \
        --runThreadN ${CORES} \
        --genomeDir ${REFS}/star \
        --readFilesIn ${R1} ${R2} \
        --readFilesCommand gunzip -c \
        --outFileNamePrefix ${ALIGNDATA}/bams/${BNAME} \
        --outSAMtype BAM SortedByCoordinate 

  done
  
# Move the log files into their own folder
mv ${ALIGNDATA}/bams/*out ${ALIGNDATA}/logs
mv ${ALIGNDATA}/bams/*tab ${ALIGNDATA}/logs

# Fastqc and indexing
for BAM in ${ALIGNDATA}/bams/*.bam
 do
   fastqc -t ${CORES} -f bam_mapped -o ${ALIGNDATA}/FastQC --noextract ${BAM}
   samtools index ${BAM}
done
