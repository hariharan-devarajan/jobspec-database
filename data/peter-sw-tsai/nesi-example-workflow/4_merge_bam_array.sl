#!/bin/bash -e

#SBATCH -A               xxxxxxxx
#SBATCH --job-name       mergeBams
#SBATCH --cpus-per-task  8
#SBATCH --time           4:00:00
#SBATCH --mem            4G
#SBATCH -o               logs/merge/merge_%a.out
#SBATCH -e               logs/merge/merge_%a.err

module load SAMtools/1.19-GCC-12.3.0

OUTDIR=merged_bam
mkdir -p $OUTDIR

FILES=($(ls -1 sample_list/*.txt))

BAMLIST=${FILES[$SLURM_ARRAY_TASK_ID]}
OUTBAM=`basename ${BAMLIST%.txt}.bam`

samtools merge -@ $SLURM_CPUS_PER_TASK -b $BAMLIST -o $OUTDIR/$OUTBAM