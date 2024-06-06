#!/bin/bash
#PBS -N test_scripts
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=480:00:00
#PBS -l mem=20gb
#PBS -M hmcqueary@uga.edu
#PBS -m ae

basedir="/scratch/hcm14449/DC_Feb2019/GC/fastqsGC"
THREADS=4
fastqc_module="FastQC/0.11.8-Java-1.8.0_144"
trimgalore_module="Trim_Galore/0.4.5-foss-2016b"
######### GC LINES ##############
GC_data_dir="/scratch/hcm14449/DC_Feb2019/GC/fastqsGC"
# mkdir /scratch/hcm14449/DC_Feb2019/GC/fastqsGC/fastQC
GC_fastQC_dir= "/scratch/hcm14449/DC_Feb2019/GC/fastqsGC/fastQC"
# ref_genome="/project/dwhlab/Holly/SC_RNAseq/RNA_seq/November_2017_Assembly/SCerRefGenome/genome.fa"
GC_trimmed_dir="/scratch/hcm14449/DC_Feb2019/GC/trim"

#run FASTQC on samples to make sure things look good and I don't need to remove any adapters or anything
cd ${basedir}

# module load ${fastqc_module}
#
# for file in $GC_data_dir/*.fastq
#
# do
#
# FBASE=$(basename $file .fastq)
# BASE=${FBASE%.fastq}
#
# fastqc $GC_data_dir/${BASE}.fastq -o $GC_fastQC_dir
#
# done
#
# module unload ${fastqc_module}

#### trimgalore

mkdir ${GC_trimmed_dir}
module load ${trimgalore_module}

for file in $GC_data_dir/*.fastq

do

FBASE=$(basename $file .fastq)
BASE=${FBASE%.fastq}

trim_galore --phred33 -q 20 -o $GC_trimmed_dir ${BASE}.fastq

done

module unload ${trimgalore_module}

### run fastQC on trimmed samples again
module load FastQC/0.11.8-Java-1.8.0_144

for file in /scratch/hcm14449/DC_Feb2019/GC/trim/*.fq

do

FBASE=$(basename $file .fq)
BASE=${FBASE%.fq}

fastqc /scratch/hcm14449/DC_Feb2019/GC/trim/${BASE}.fq -o /scratch/hcm14449/DC_Feb2019/GC/trim

done
