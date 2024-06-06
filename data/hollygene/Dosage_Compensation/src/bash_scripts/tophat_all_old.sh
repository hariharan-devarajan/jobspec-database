#!/bin/bash
#PBS -N test.sample1
#PBS -q batch
#PBS -l nodes=1:ppn=4:AMD
#PBS -l walltime=96:00:00
#PBS -l pmem=100gb
#PBS -M hmcqueary@uga.edu
#PBS -m ae

cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/MA_old/fastq/
module load tophat/2.1.1

for file in ./*.fastq

do

FBASE=$(basename $file .fastq)
BASE=${FBASE%.fastq}
THREADS=4
time  tophat -p $THREADS -o /lustre1/hcm14449/SC_RNAseq/RNA_seq/MA_old/tophat/${BASE}_tophat_out \
-i 10 -I 1000 \
--transcriptome-index=/lustre1/hcm14449/SC_RNAseq/RNA_seq/tophat/transcriptome_data/known \
/lustre1/hcm14449/SC_RNAseq/RNA_seq/tophat/genome \
/lustre1/hcm14449/SC_RNAseq/RNA_seq/MA_old/fastq/${BASE}.fastq

done
