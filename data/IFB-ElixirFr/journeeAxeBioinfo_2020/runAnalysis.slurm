#!/bin/bash
#
#SBATCH --mem 50GB
#SBATCH --cpus-per-task 30
#SBATCH --partition fast

module load snakemake fastqc bowtie2 samtools subread slurm-drmaa

snakemake --drmaa --jobs=$SLURM_CPUS_PER_TASK -s demo.smk --configfile config.yml

singularity exec -B $PWD:/home/ journee_axe_bioinfo_latest.sif Rscript --vanilla sartools.R