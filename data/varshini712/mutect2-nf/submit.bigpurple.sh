#!/bin/bash

#SBATCH -o slurm-%j.out
#SBATCH -J NGS580-GATK4_test
#SBATCH -p intellispace
#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH --export=HOSTNAME

./nextflow run main.nf
