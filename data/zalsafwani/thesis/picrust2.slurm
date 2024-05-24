#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64000
#SBATCH --time=12:00:00
#SBATCH --job-name=picrust_analysis
#SBATCH --partition=batch,guest
#SBATCH --error=/work/biocore/zalsafwani/CRC_data/raw_reads/qiime2/second_analysis/picrust/script_output/err.out
#SBATCH --output=/work/biocore/zalsafwani/CRC_data/raw_reads/qiime2/second_analysis/picrust/script_output/std.out
#SBATCH --open-mode=append


cd /work/biocore/zalsafwani/CRC_data/raw_reads/qiime2/second_analysis/picrust


module load picrust2/2.4

sed '1d' feature-table.tsv > fixed-feature-table.tsv
picrust2_pipeline.py -s ../../exports/rep-seqs/dna-sequences.fasta -i fixed-feature-table.tsv -o results
