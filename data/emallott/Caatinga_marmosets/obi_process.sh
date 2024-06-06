#!/bin/bash

#SBATCH -J obi_process
#SBATCH -A b1057
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elizabeth.mallott@northwestern.edu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH --output=/home/ekm9460/obi_process_plant.out
#SBATCH --error=/home/ekm9460/obi_process_plant.err
#SBATCH -p b1057

module purge all

module load python

source activate /home/ekm9460/obi3-env

PROJECT="/projects/b1057/liz/tiputini_caatinga_diet/"

for aligned in ${PROJECT}/plant/*
  do 
     obi alignpairedend -R $aligned/reads1 $aligned/reads2 $aligned/aligned_reads
done

for paired in ${PROJECT}/plant/*
  do
     obi grep -a mode:alignment $paired/aligned_reads $paired/good_sequences
done

for goodseq in ${PROJECT}/plant/*
  do
     obi ngsfilter -e  -t $goodseq/ngsfile --no-tags -u $goodseq/unidentified_sequences $goodseq/good_sequences $goodseq/identified_sequences
done

