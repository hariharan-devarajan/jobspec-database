#!/bin/bash

#SBATCH --mail-type=end
#SBATCH --mail-user=zly18810602991@163.com
#SBATCH --job-name=test
#SBATCH --partition=small
#SBATCH -N 1
#SBATCH -J Caveman
#SBATCH --ntasks-per-node=8
#SBATCH --output=%j.o
#SBATCH --error=%j.e



# Initialize a variable with an intuitive name to store the name of the input fastq file
tumor=$1
normal=$2

# Data from Pfizer
baseSuper=`basename ${tumor} .sort.bam`
base=$(echo ${baseSuper} | cut -d'_' -f 2)  # get the filename with basename function
echo "base is ${base}"
echo "Tumor sample is ${tumor}"
echo "Normal sample is ${normal}"

# Set working dir and annotation dir
project_dir=/lustre/home/acct-medkwf/medkwf4/results/MRD/CC_data/${base}
mkdir -p ${project_dir}
working_dir=${project_dir}/Caveman
mkdir -p ${working_dir}
annovar=${project_dir}/ANNOVAR
mkdir -p ${annovar}

# Directory with reference genome
genome_19=/lustre/home/acct-medkwf/medkwf/database/GATK/hg19/ucsc.hg19.fasta
#genome_19=/lustre/home/acct-medkwf/medkwf4/reference/GRCh37_ucsc/hg19.fa
introns_bed=/lustre/home/acct-medkwf/medkwf4/reference/bed/UCSC_Introns.tsv

# Set up environment
#source activate dna
export IAMGE_NAME=/lustre/home/acct-medkwf/medkwf4/software/CRUK/dockstore-cgpwxs_3.1.7.sif
export IAMGE_NAME_WRAPPER=/lustre/home/acct-medkwf/medkwf4/software/CRUK/docker-caveman_v1.0.0.sif
#export IAMGE_NAME=/lustre/home/acct-medkwf/medkwf4/software/CRUK/docker-caveman_v1.0.0.sif

# specify the number of cores to use
cores=4

# Set output files

# Use Caveman
# -g  --ignore-regions-file [file]    Location of tsv ignore regions file
#touch ${working_dir}/splitList
#singularity exec $IAMGE_NAME caveman setup \
#-t ${tumor} \
#-n ${normal} \
#-r ${genome_19}.fai \
#-g ${introns_bed} \
#-f ${working_dir} \
#-c ${working_dir}/caveman.cfg.ini \
#-l ${working_dir}/splitList \
#-a ${working_dir}/alg_bean
#
##not split chromosomes correctly
#singularity exec $IAMGE_NAME caveman split \
#-i 2 \
#-f ${working_dir}/caveman.cfg.ini \
#-e 100000 \

#singularity exec $IAMGE_NAME caveman mstep \
#-i 2 \
#-f ${working_dir}/caveman.cfg.ini
#
#singularity exec $IAMGE_NAME caveman merge \
#-f ${working_dir}/caveman.cfg.ini
#
#singularity exec $IAMGE_NAME caveman estep \
#-i 2 \
#-f ${working_dir}/caveman.cfg.ini

# Use Caveman Wrapper
#-tc ${working_dir}/H016_tumor.cn.bed \
#-nc ${working_dir}/H016_normal.cn.bed \
#singularity run $IAMGE_NAME_WRAPPER \
singularity exec $IAMGE_NAME caveman.pl \
-o ${working_dir} \
-r ${genome_19}.fai \
-tb ${tumor} \
-nb ${normal} \
-ig ${introns_bed} \
-td 1 \
-nd 1 \
-species human \
-sa GRCh37d5 \
-st genomic \
-flag-bed-files /lustre/home/acct-medkwf/medkwf4/reference/dbsnp/dbSnp153Common.bed \
-germline-indel ${working_dir}/test.bed \
-unmatched-vcf ${working_dir}/unmatched-vcf \
-threads 8 \
-normal-protocol WXS \
-tumour-protocol WXS
 

