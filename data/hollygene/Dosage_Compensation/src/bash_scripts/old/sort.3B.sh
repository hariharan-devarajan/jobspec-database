#!/bin/bash

#PBS -N sort_3B
#PBS -q batch
#PBS -l nodes=1:ppn=5:AMD
#PBS -l mem=5gb
#PBS -l walltime=480:00:00
#PBS -M hmcqueary@uga.edu
#PBS -m ae
#PBS -o /lustre1/hcm14449/SC_RNAseq/sort_3.o
#PBS -e /lustre1/hcm14449/SC_RNAseq/sort_3.e

module load samtools/1.3.1
cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/tophat/old

time samtools sort -m 100G -o /lustre1/hcm14449/SC_RNAseq/RNA_seq/tophat/old/SORTHolly_3B_S8_R1_001_tophat_out/Holly_3B_S8_R1_001_tophat_out.sorted.bam \
-T /lustre1/hcm14449/SC_RNAseq/RNA_seq/tophat/old/Holly_3B_S8_R1_001_tophat_out/accepted_hits \
/lustre1/hcm14449/SC_RNAseq/RNA_seq/tophat/old/Holly_3B_S8_R1_001_tophat_out/accepted_hits.bam
