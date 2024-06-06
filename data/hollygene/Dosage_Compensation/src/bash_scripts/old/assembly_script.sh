#!/bin/bash
#PBS -N assembly_test
#PBS -q batch
#PBS -l nodes=1:ppn=4:AMD
#PBS -l walltime=480:00:00
#PBS -l mem=100gb
#PBS -M hmcqueary@uga.edu
#PBS -m ae

basedir="/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly"
cd $basedir
THREADS=4


#run FASTQC on samples to make sure things look good and I don't need to remove any adapters or anything
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC
#trying to rename files so that I can eventually run a loop with cuffdiff
#rename 'GC' A *.fq
#mkdir fastqc
#module load java/jdk1.8.0_20 fastqc
#fastqc *.fastq -o fastqc

#module unload java/jdk1.8.0_20 fastqc

#trim sequences a little
#trying trimgalore since trimmomatic didnt seem to work
#module load trimgalore/0.4.4

#for file in ./*.fastq

#do

#FBASE=$(basename $file .fastq)
#BASE=${FBASE%.fastq}

#trim_galore --phred33 -q 20 -o trimmed ${BASE}.fastq

#done

#module unload trimgalore/0.4.4

#map to reference using bowtie2
#module load bowtie2/2.2.9
#build index for ref genome
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/
#bowtie2-build genome.fa genome
#bowtie2-inspect -s genome
#map sequences for line 2 first
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/trimmed
#bowtie2 -x /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome -U Holly_2A_S1_R1_001_trimmed.fq -S Holly_2A.sam
#bowtie2 -x /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome -U Holly_2B_S7_R1_001_trimmed.fq -S Holly_2B.sam
#bowtie2 -x /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome -U Holly_2C_S4_R1_001_trimmed.fq -S Holly_2C.sam
#bowtie2 -x /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome -U Holly-GC-7-redo_S4_R1_001_trimmed.fq -S Holly_7A.sam
#bowtie2 -x /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome -U Holly_7B_S12_R1_001_trimmed.fq -S Holly_7B.sam
#bowtie2 -x /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome -U Holly_7C_S9_R1_001_trimmed.fq -S Holly_7C.sam
#for file in ./*.fq
#do
#FBASE=$(basename $file .fq)
#BASE=${FBASE%.fq}
#bowtie2 -x /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome \
#-U ${BASE}.fq -S ${BASE}.sam
#done
#module unload bowtie2/2.2.9
#module load samtools/1.3.1
#turn sam files into bam files
#samtools view -bS -T  Holly_2A_S1_R1_001_trimmed.sam -o Holly_2A_S1_R1_001_trimmed.bam
#samtools view -bS Holly_2B_S7_R1_001_trimmed.sam -o Holly_2B_S7_R1_001_trimmed.bam
#samtools view -bS Holly_2C_S4_R1_001_trimmed.sam -o Holly_2C_S4_R1_001_trimmed.bam
#samtools view -b Holly_7B_S12_R1_001_trimmed.sam
#samtools view -b Holly_7C_S9_R1_001_trimmed.sam
#samtools view -b Holly-A-7-redo_S4_R1_001_trimmed.sam
#sort the bam files
#samtools sort -m 5G -@ $THREADS -O bam -T Holly_2A.tmp Holly_2A.bam > Holly_2A.sort.bam
#samtools sort -m 5G -@ $THREADS -O bam -T Holly_2B.tmp Holly_2B.bam > Holly_2B.sort.bam
#samtools sort -m 5G -@ $THREADS -O bam -T Holly_2C.tmp Holly_2C.bam > Holly_2C.sort.bam
#samtools sort -m 5G -@ $THREADS -O bam -T Holly_7A.tmp Holly_7A.bam > Holly_7A.sort.bam
#samtools sort -m 5G -@ $THREADS -O bam -T Holly_7B.tmp Holly_7B.bam > Holly_7B.sort.bam
#samtools sort -m 5G -@ $THREADS -O bam -T Holly_7C.tmp Holly_7C.bam > Holly_7C.sort.bam
#module unload samtools/1.3.1
#load in bwa for mapping
#module load bwa/0.7.15
#bwa index genome.fa
#map reads to ref genome using bwa
#bwa mem Holly_2A_S1_R1_001.fastq
#Holly_2B_S7_R1_001.fastq
#Holly_2C_S4_R1_001.fastq

#load in tophat
#module load tophat/2.1.1

#index transcriptome file
#tophat -G genes.gtf --transcriptome-index=transcriptome_data/known genome

#run tophat on all samples
#do GC ones first
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/trimmed
#mkdir "/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/trimmed/tophat_test/"

#for file in ./*.fq

#do

#FBASE=$(basename $file .fq)
#BASE=${FBASE%.fq}
#tophat -p $THREADS -o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/trimmed/tophat_test/${BASE}_tophat_out \
#-i 10 -I 1000 \
#--transcriptome-index=/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/transcriptome_data/known \
#/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome \
#./${BASE}.fq

#done

#unload tophat
#module unload tophat/2.1.1

#load in cufflinks
module load cufflinks/2.2.1

#then run cufflinks on all GC samples
cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/trimmed/tophat_test

ls >> output.txt
mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/Cufflinks
cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/Cufflinks

#run cufflinks on all samples
while read SampleName
do
mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/Cufflinks_test/cufflinks${SampleName}

cufflinks \
-p $THREADS
-g /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genes.gtf \
-o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/Cufflinks/cufflinks${SampleName} \
-b /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome.fa \
/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/trimmed/tophat_test/${BASE}_tophat_out

done < /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/trimmed/tophat_test/output.txt

module unload cufflinks/2.2.1

#mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/quant_test
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/quant_test
#run cuffquant to estimate expression levels to put into cuffdiff

#while read SampleName
#do
#mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/quant_test/cuffquant${SampleName}

#cuffquant \
#-p $THREADS
#-g /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genes.gtf \
#-o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/quant_test/cuffquant${SampleName} \
#-b /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome.fa \
#/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/tophat_test/${BASE}_tophat_out/accepted_hits.bam

#done < /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/GC/tophat_test/output.txt

#module unload cufflinks/2.2.1

#####################################################################################################################
#then MA new

cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new
#trying to rename files so that I can eventually run a loop with cuffdiff
#rename 'GC' A *.fq
#mkdir fastqc
#module load java/jdk1.8.0_20 fastqc
#fastqc *.fastq -o fastqc

#module unload java/jdk1.8.0_20 fastqc

#trim sequences a little
#trying trimgalore since trimmomatic didnt seem to work
#module load trimgalore/0.4.4
#mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/trimmed
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new

#for file in ./*.fastq

#do

#FBASE=$(basename $file .fastq)
#BASE=${FBASE%.fastq}

#trim_galore --phred33 -q 20 -o trimmed ${BASE}.fastq

#done

#module unload trimgalore/0.4.4

module load tophat/2.1.1
cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/trimmed
mkdir "/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/trimmed/tophat/"

for file in ./*.fq

do

FBASE=$(basename $file .fq)
BASE=${FBASE%.fq}
tophat -p $THREADS -o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/trimmed/tophat/${BASE}_tophat_out \
-i 10 -I 1000 \
--transcriptome-index=/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/transcriptome_data/known \
/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome \
./${BASE}.fq

done

#unload tophat
module unload tophat/2.1.1
#load in tophat
#module load tophat/2.1.1
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new

#for file in ./*.fastq

#do

#FBASE=$(basename $file .fastq)
#BASE=${FBASE%.fastq}

#tophat -p $THREADS -o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/tophat/${BASE}_tophat_out -i 10 -I 1000 \
#--transcriptome-index=transcriptome_data/known \
#genome \
#./${BASE}.fastq

#done
#unload tophat
#module unload tophat/2.1.1

#load in cufflinks
#module load cufflinks/2.2.1
#then run cufflinks on all new MA samples
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/tophat
#printf '%s\n' * > output.txt

#run cufflinks on all samples
#while read SampleName
#do
#mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/Cufflinks/cufflinks${SampleName}
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/Cufflinks
#cufflinks \
#-p $THREADS
#-g /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genes.gtf \
#-o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/Cufflinks/cufflinks${SampleName} \
#-b /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome.fa \
#/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/tophat/${BASE}_tophat_out

#done < /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/tophat/output.txt

#run cuffquant to estimate expression levels to put into cuffdiff
#while read SampleName
#do
#mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/quant/cuffquant${SampleName}
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/quant
#cuffquant \
#-p $THREADS
#-g /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genes.gtf \
#-o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/quant/cuffquant${SampleName} \
#-b /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome.fa \
#/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/tophat/${BASE}_tophat_out/accepted_hits.bam

#done < /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_new/tophat/output.txt

#unload cufflinks
#module unload cufflinks/2.2.1

#####################################################################################################################
#then MA old
#load in tophat
#module load tophat/2.1.1
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old

#for file in ./*.fastq

#do

#FBASE=$(basename $file .fastq)
#BASE=${FBASE%.fastq}

#tophat -p $THREADS -o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/tophat/${BASE}_tophat_out -i 10 -I 1000 \
#--transcriptome-index=transcriptome_data/known \
#genome \
#./${BASE}.fastq

#done
#unload tophat
#module unload tophat/2.1.1

#load cufflinks
#module load cufflinks/2.2.1

#then run cufflinks on all new MA samples
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/tophat
#printf '%s\n' * > output.txt

#run cufflinks on all samples
#while read SampleName
#do
#mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/Cufflinks/cufflinks${SampleName}
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/Cufflinks
#cufflinks \
#-p $THREADS
#-g /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genes.gtf \
#-o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/Cufflinks/cufflinks${SampleName} \
#-b /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome.fa \
#/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/tophat/${BASE}_tophat_out

#done < /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/tophat/output.txt

#run cuffquant to estimate expression levels to put into cuffdiff
#while read SampleName
#do
#mkdir /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/quant/cuffquant${SampleName}
#cd /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/quant
#cuffquant \
#-p $THREADS
#-g /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genes.gtf \
#-o /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/quant/cuffquant${SampleName} \
#-b /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/genome.fa \
#/lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/tophat/${BASE}_tophat_out/accepted_hits.bam

#done < /lustre1/hcm14449/SC_RNAseq/RNA_seq/November_2017_Assembly/MA_old/tophat/output.txt

#run cuffdiff to find differential expression between each line and its ancestor
#cuffdiff -p $THREADS --library-type fr-unstranded -o FNR_cuffdiff --labels wild-type,dFNR MG1655.ref.gtf SRR5344681_cuffquant/abundances.cxb,SRR5344682_cuffquant/abundances.cxb SRR5344683_cuffquant/abundances.cxb,SRR5344684_cuffquant/abundances.cxb

#unload cufflinks
#module unload cufflinks/2.2.1
