#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --array=1-14%14

## DESCRIPTION: This array job maps reads to the human ref genome.

echo "START"
date

cd /data/zhanglab/zpimentel/ChIP-Seq/02_mapping
module load Bowtie2/2.4.2-GCC-9.3.0
module load SAMtools/1.9-foss-2018b
module load BEDTools/2.30.0-GCC-10.2.0

# Set up array job to download the reads at the same time
# ID is now the SRA ID's we need to download
ID=$(head -n $SLURM_ARRAY_TASK_ID ../sample.list | tail -n 1 | cut -f1)

# map the reads
bowtie2 --threads 9 -x GRCh38 -U ../01_raw_data/${ID}_1.fastq -S ./mapping_out/${ID}.sam --no-unal --very-sensitive-local

# convert sam to bam
samtools view -S -b ./mapping_out/${ID}.sam > ./mapping_out/${ID}.bam

# sort bam files
samtools sort ./mapping_out/${ID}.bam -o ./mapping_out/${ID}_sorted.bam

# covert to bed file
bamToBed -i ./mapping_out/${ID}_sorted.bam > ./mapping_out/${ID}.bed

echo "DONE"
date
