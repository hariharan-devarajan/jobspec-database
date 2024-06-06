#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH -p Lewis
#SBATCH -N 1                            # Reserve this number of nodes
#SBATCH --cpus-per-task 16              # Reserve this number of cpus
#SBATCH --mem=50G                       # Reserve this amount of memory
#SBATCH -t 0-10:00                      # Time Reservation (days-hours:minutes)
#
#
## labels and outputs
#SBATCH -J nf-custom-gat4k  # job name - shows up in sacct and squeue
#SBATCH -o nf_custom_gatk4-%j.out  # filename for the output from this job (%j = job#)
#
#-------------------------------------------------------------------------------

#module load nextflow/21.04.1
#module load gatk/4.1.8.1
#module load bwa/0.7.17
#module load picard/2.23.8
#module load snpeff/5.0
#module load singularity/singularity

nextflow run KyleStiers/variant-calling-pipeline-gatk4 -with-singularity KyleStiers/variant-calling-pipeline-gatk4
