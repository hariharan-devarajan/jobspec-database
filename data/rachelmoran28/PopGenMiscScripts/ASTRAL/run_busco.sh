#!/bin/bash
#SBATCH -p small,amdsmall
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=62gb
#SBATCH -t 80:00:00
#SBATCH -J BUSCO


module load python/2

src=/home/research/genome/assemblies_configs/

input=/home/research/genome/final_assembly/pilon_final_122418.fa  #genome assembly fasta
lineage=/home/research/genome/assemblies_configs/lineage/actinopterygii_odb9 #Visit http://busco.ezlab.org for lineages - pick one that's closest to your study organism and save somewhere in your h
busco=/home/software/busco/2.0/bin/BUSCO.py #this is the correct path for BUSCO v2.0  


#usage: python BUSCO.py -i [SEQUENCE_FILE] -l [LINEAGE] -o [OUTPUT_NAME] -m [MODE] [OTHER OPTIONS]
#choose a species (-sp) that is a close relative of your study orgnaism; list available here: /panfs/roc/msisoft/augustus/3.2.3.CentOS7/config/species


python2 $busco -i $input -o 122418_final_assembly -l $lineage -f -m geno -sp zebrafish

