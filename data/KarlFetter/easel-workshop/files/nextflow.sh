#!/bin/bash
#SBATCH --job-name=easel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --mem=10G
#SBATCH --mail-user=

module load nextflow
source activate envAGAT

nextflow run main.nf -w karl -with-report -with-timeline -with-dag acer_nf.png \
--species acer_negundo \
--genome /core/labs/Wegrzyn/easel-workshop/data/genome/chr1.fna \
--outdir . \
--sra /core/labs/Wegrzyn/easel-workshop/data/sra/acer_negundo.txt
