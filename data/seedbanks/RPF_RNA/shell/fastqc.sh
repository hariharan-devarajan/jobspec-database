#!/bin/bash
#PBS -k o
#PBS -l nodes=1:ppn=8,vmem=100gb,walltime=10:00:00
#PBS -M wrshoema@indiana.edu
#PBS -m abe
#PBS -j oe
module load java
module load fastqc
module load bowtie2
module load tophat
module load python
module load gcc
module load cutadapt

# Index reference
mkdir -p /N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reference/KBS0714/bwt

bowtie2-build /N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reference/KBS0714/prokka/G-Chr1.fna \
    /N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reference/KBS0714/bwt/G-Chr1

# Asess quality of FASTQ files
mkdir -p /N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reads_quality

for file in /N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reads_raw/*fastq.gz
do
  fastqc "$file" --outdir=/N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reads_quality
done

# Trim the reads
mkdir -p /N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reads_clean

for IN in /N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reads_raw/*fastq.gz
do
  name="$(echo "$IN" | cut -d "/" -f9 | cut -d "." -f 1)"
  OUT="/N/dc2/projects/Lennon_Sequences/2016_RPF_RNA/data/reads_clean/${name}_cleaned.fastq.gz"
  cutadapt -q 30 -u 10 -o $OUT $IN
done
