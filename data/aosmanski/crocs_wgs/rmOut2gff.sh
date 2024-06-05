#!/bin/bash
#SBATCH --job-name=mask
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=nocona
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -a 1-20
#SBATCH --mem-per-cpu=32G

module load gcc/10.1.0 bedtools2/2.29.2

#For some reason the perl pipeline isn't working well with a large repeatmasker output dataset.
#Below is some code which breaks up the RMout file by "query sequence"

RMPATH=/lustre/work/daray/software/RepeatMasker-4.1.0/util
NAMESFILE=/lustre/scratch/aosmansk/new_croc_assemblies/repeatmasker/RM_LIST
CHRANGE=$(awk "NR==$SLURM_ARRAY_TASK_ID" $NAMESFILE)
GENOME=${CHRANGE}_ref-based_assembly
BASEDIR=/lustre/scratch/aosmansk/new_croc_assemblies/repeatmasker
RMOUT=${GENOME}.fa.out.new
WORKDIR=/lustre/scratch/aosmansk/new_croc_assemblies/repeatmasker/${CHRANGE}"_RM"

#First, create a working directory for each species
cd $WORKDIR
mkdir $WORKDIR/rmout2gff
DIR=$WORKDIR/rmout2gff

#gunznip the RMout file.
#Copy the newly unzipped file and zip a copy back together to keep some raw data "protected"
#mv ${GENOME}.fa.out ${GENOME}.fa.out.old
#gunzip ${GENOME}.fa.out.gz
#cp ${GENOME}.fa.out ${GENOME}.fa.out.new
#gzip ${GENOME}.fa.out

#Identify the names of all the query sequences in the RMOUT file.
awk '{print $5}' $WORKDIR/$RMOUT | uniq | sed -e '1,3d' > $DIR/QUERIES.txt

#Jump into the rmout2gff directory
cd $DIR

#Seperate the RMout file by query sequence (column 5) and convert to gff
echo "##gff-version 3" >> ${CHRANGE}.gff
for line in $(cat $DIR/QUERIES.txt); do \
        awk -v var="$line" '$5==var' $WORKDIR/$RMOUT > $line.rmout; \
        min=1; \
        max=$(awk 'NR==1{max = $7 + 0; next} {if ($7 > max) max = $7;} END {print max}' $line.rmout); \
        echo "##sequence-region" $line $min $max >> ${CHRANGE}.gff; \
        awk '{printf $5"\t""RepeatMasker""\t""dispersed_repeat""\t"$6"\t"$7"\t"$1"\t";if ($9 =="C") print "-""\t"".""\t""Target="$10,$14,$13; else print "+""\t"".""\t""Target="$10,$12,$13}' $line.rmout >> ${CHRANGE}.gff; \
        rm $line.rmout; \
done
