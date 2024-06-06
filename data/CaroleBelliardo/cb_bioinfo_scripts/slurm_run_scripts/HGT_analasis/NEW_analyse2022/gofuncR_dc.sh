#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=8     # number of CPU per task
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per Nodes
#SBATCH -J "gofunc"   # job name
#SBATCH --mail-user=carole.belliardo@inra.fr   # email address
#SBATCH --mail-type=ALL
#SBATCH -e slurm-go-%j.err
#SBATCH -o slurm-go-%j.out
#SBATCH -p all

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load singularity/3.5.3 


IMG='/lerins/hub/projects/25_tools/GOfuncR/GOFunc.sif'


singularity run -B "/lerins/hub" -B "/work/$USER" $IMG snakemake --snakefile /lerins/hub/DB/WORKFLOW/GOfuncR/Snakefile -j $SLURM_CPUS_PER_TASK --configfile ${path}/param.yaml


