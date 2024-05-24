#!/bin/bash

# 1. Configure resources
#SBATCH -A punim0930
#SBATCH --partition=physical
#SBATCH --job-name="bioinf benchmark" 
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --mem=32G

# 2. Set the working dir
WORK_DIR=/scratch/punim0930/evan/bioinf-io-benchmark
cd $WORK_DIR

# 3. Makesure software is in your path
module load BWA/0.7.17-intel-2018.u4 SAMtools/1.9-intel-2018.u4 Trimmomatic/0.36-Java-1.8.0_152
TRIMMOMATIC=/usr/local/easybuild/software/Trimmomatic/0.36/

# 4. The reference genome
REFERENCE=$WORK_DIR/hg38.fa

# 5. Sequence data
READ1=$WORK_DIR/fake_data1.fq
READ2=$WORK_DIR/fake_data2.fq


# Trim
echo trim started
time java -jar ${TRIMMOMATIC}/trimmomatic-0.36.jar PE -threads `nproc` $READ1 $READ2 -baseout ${WORK_DIR}/output-trimmed.fastq.gz ILLUMINACLIP:${TRIMMOMATIC}/adapters/TruSeq3-PE.fa:1:30:20:4:true
echo trim ended

# Align
echo align started
time bwa mem -M -t `nproc` $REFERENCE ${WORK_DIR}/output-trimmed_1P.fastq.gz ${WORK_DIR}/output-trimmed_2P.fastq.gz > ${WORK_DIR}/aln.sam
echo align ended

# Sort
echo sort started
time samtools sort -@ `nproc` ${WORK_DIR}/aln.sam > ${WORK_DIR}/aln.bam
echo sort ended

# index
echo index started
time samtools index ${WORK_DIR}/aln.bam
echo index ended
