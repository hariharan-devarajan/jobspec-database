#!/bin/bash

#SBATCH --job-name=process_rna
#SBATCH --output=process_rna.txt
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=
#SBATCH --partition=pi_krishnaswamy

module load miniconda
conda activate splicenn

python process_rna_seq.py

