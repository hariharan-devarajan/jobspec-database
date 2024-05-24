#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64000
#SBATCH --time=12:00:00
#SBATCH --job-name=picrust_analysis
#SBATCH --partition=batch,guest
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maiabennett@unomaha.edu
#SBATCH --error=/common/claytonlab/maibennett/gorilla_cortisol/AnimalName-IndClusters/Cenzoo/qiime2/script_output/errpi.out
#SBATCH --output=/common/claytonlab/maibennett/gorilla_cortisol/AnimalName-IndClusters/Cenzoo/qiime2/script_output/stdpi.out
#SBATCH --open-mode=append


cd /common/claytonlab/maibennett/gorilla_cortisol/AnimalName-IndClusters/Cenzoo/qiime2/picrust


module load picrust2/2.4

#sed '1d' /common/claytonlab/maibennett/gorilla_cortisol/AnimalName-IndClusters/Cenzoo/qiime2/exports/table/feature-table-w-taxa.biom > fixed-feature-table.tsv
picrust2_pipeline.py -s /common/claytonlab/maibennett/gorilla_cortisol/AnimalName-IndClusters/Cenzoo/qiime2/exports/rep-seqs/dna-sequences.fasta -i /common/claytonlab/maibennett/gorilla_cortisol/AnimalName-IndClusters/Cenzoo/qiime2/exports/table/feature-table-w-taxa.biom -o results
