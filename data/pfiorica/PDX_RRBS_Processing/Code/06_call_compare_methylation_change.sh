#!/bin/bash
#SBATCH --partition=general-compute --qos=general-compute
#SBATCH --time=24:00:00
#SBATCH --job-name=compare_methylation_change_plots
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32000
##SBATCH --requeue
#SBATCH --output=logs_bismark/%x.%j.out
#SBATCH --error=logs_bismark/%x.%j.err
#module load gcc
module load R/3.5.1
cd /projects/rpci/joyceohm/pnfioric/PDX_RRBS_Processing/Code

Rscript 06_Compare_Methylation_Change.R
