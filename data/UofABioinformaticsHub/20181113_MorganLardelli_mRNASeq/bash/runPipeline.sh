#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH -o /data/biohub/20181113_MorganLardelli_mRNASeq/slurm/%x_%j.out
#SBATCH -e /data/biohub/20181113_MorganLardelli_mRNASeq/slurm/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stephen.pederson@adelaide.edu.au

## Cores
CORES=16

######################################
## Full run of the data from SAHMRI ##
######################################

## Modules
module load FastQC/0.11.7
module load STAR/2.5.3a-foss-2016b
module load SAMtools/1.3.1-GCC-5.3.0-binutils-2.25
module load AdapterRemoval/2.2.1-foss-2016b
module load GCC/5.4.0-2.26

## Biohub/local
featureCounts=/data/biohub/local/subread-1.5.2-Linux-x86_64/bin/featureCounts

## Genomic Data Files
REFS=/data/biorefs/reference_genomes/ensembl-release-94/danio-rerio/
GTF=${REFS}/Danio_rerio.GRCz11.94.chr.gtf

# # The STAR genome index was generated manually on clarence after mounting /data/biorefs
# # STAR 2.6.0a was used for generation of the index, whilst STAR 2.5.3 will be 
# # used for alignment
# STAR \
#   --runThreadN 20 \
#   --runMode genomeGenerate \
#   --genomeDir ${REFS}/star/ \
#   --genomeFastaFiles ${REFS}/Danio_rerio.GRCz11.dna.primary_assembly.fa \
#   --sjdbGTFfile ${REFS}/Danio_rerio.GRCz11.94.chr.gtf \
#   --sjdbOverhang 74

## Directories
PROJROOT=/data/biohub/20181113_MorganLardelli_mRNASeq

## Directories for Initial FastQC
RAWDATA=${PROJROOT}/0_rawData
mkdir -p ${RAWDATA}/FastQC

## Setup for Trimmed data
TRIMDATA=${PROJROOT}/1_trimmedData
mkdir -p ${TRIMDATA}/fastq
mkdir -p ${TRIMDATA}/FastQC
mkdir -p ${TRIMDATA}/logs

## Setup for genome alignment
ALIGNDATA=${PROJROOT}/2_alignedData
mkdir -p ${ALIGNDATA}/logs
mkdir -p ${ALIGNDATA}/bams
mkdir -p ${ALIGNDATA}/FastQC
mkdir -p ${ALIGNDATA}/featureCounts


##--------------------------------------------------------------------------------------------##
## FastQC on the raw data
##--------------------------------------------------------------------------------------------##

fastqc -t ${CORES} -o ${RAWDATA}/FastQC --noextract ${RAWDATA}/fastq/*fq.gz

##--------------------------------------------------------------------------------------------##
## Trimming the Merged data
##--------------------------------------------------------------------------------------------##

for R1 in ${RAWDATA}/fastq/*R1.fq.gz
  do

    echo -e "Currently working on ${R1}"

    # Now create the output filenames
    out1=${TRIMDATA}/fastq/$(basename $R1)
    BNAME=${TRIMDATA}/fastq/$(basename ${R1%_R1.fq.gz})
    echo -e "Output file 1 will be ${out1}"

    echo -e "Trimming:\t${R1}"
    # Trim
    AdapterRemoval \
      --adapter1 AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC \
      --gzip \
      --trimns \
      --trimqualities \
      --minquality 20 \
      --minlength 35 \
      --threads ${CORES} \
      --basename ${BNAME} \
      --output1 ${out1} \
      --file1 ${R1} 

  done

# Move the log files into their own folder
mv ${TRIMDATA}/fastq/*settings ${TRIMDATA}/logs

# Run FastQC
fastqc -t ${CORES} -o ${TRIMDATA}/FastQC --noextract ${TRIMDATA}/fastq/*R1.fq.gz


##--------------------------------------------------------------------------------------------##
## Aligning trimmed data to the genome
##--------------------------------------------------------------------------------------------##

## Aligning, filtering and sorting
for R1 in ${TRIMDATA}/fastq/*R1.fq.gz
  do
  
  BNAME=$(basename ${R1%_R1.fq.gz})
  echo -e "STAR will align:\t${R1}"

    STAR \
        --runThreadN ${CORES} \
        --genomeDir ${REFS}/star \
        --readFilesIn ${R1} \
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


##--------------------------------------------------------------------------------------------##
## featureCounts 
##--------------------------------------------------------------------------------------------##

## Feature Counts - obtaining all sorted bam files
sampleList=`find ${ALIGNDATA}/bams -name "*out.bam" | tr '\n' ' '`

## Running featureCounts on the sorted bam files
${featureCounts} -Q 10 \
  -s 1 \
  -T ${CORES} \
  -a ${GTF} \
  -o ${ALIGNDATA}/featureCounts/counts.out ${sampleList}

## Storing the output in a single file
cut -f1,7- ${ALIGNDATA}/featureCounts/counts.out | \
sed 1d > ${ALIGNDATA}/featureCounts/genes.out



