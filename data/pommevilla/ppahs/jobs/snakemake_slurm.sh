#!/bin/bash
#SBATCH --job-name=ppahs
#SBATCH -p short
#SBATCH --mem=16gb
#SBATCH -n 12
#SBATCH -t 8:00:00
#SBATCH -o "logs/slurm/out/stdout.%j.%N"
#SBATCH -e "logs/slurm/err/stderr.%j.%N"
#SBATCH --mail-user=paul.villanueva@usda.gov
#SBATCH --mail-type=BEGIN,END,FAIL

cd /project/fsepru/paul.villanueva/repos/ppahs
source /home/${USER}/.bashrc
source activate snakemake


date
time snakemake --use-conda --profile workflow/profile -p 
date
