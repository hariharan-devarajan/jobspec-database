#!/bin/bash
#PBS -N log.count
#PBS -j oe
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=4Gb

module load gcc/6.2.0
module load python/2.7.13

cd /scratch/a.cri.mforde/sra_downloads

htseq-count -f bam --type=gene $input_file g38/Homo_sapiens.GRCh38.92.gtf > "$input_file".g38.count.csv

done



