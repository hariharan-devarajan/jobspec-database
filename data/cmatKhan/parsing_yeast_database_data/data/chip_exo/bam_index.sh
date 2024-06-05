#!/bin/bash

#SBATCH --job-name=bam_index
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M
#SBATCH --time=00:30:00
#SBATCH --output=bam_index.out
#SBATCH --error=bam_index.err

eval $(spack load --sh samtools@1.13)

lookup="$1"

# Extract the BAM file path for the current array index from the lookup file
BAM_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $lookup)

# Index the BAM file
samtools index $BAM_FILE

echo "BAM file $BAM_FILE indexed successfully."
