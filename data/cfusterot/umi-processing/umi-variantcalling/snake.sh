#!/bin/bash

# export SNAKEMAKE_SLURM_DEBUG=1

#SBATCH --job-name=variantcalling
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=24000M
#SBATCH --output=/data/gpfs-1/users/cofu10_c/scratch/P3473/slurm_logs/%x.%j.out
#SBATCH --error=/data/gpfs-1/users/cofu10_c/scratch/P3473/slurm_logs/%x.%j.err

snakemake \
    --nt \
    --jobs 60 \
    --cluster-config /data/gpfs-1/users/cofu10_c/work/pipelines/umi-processing/config/cluster_config.yaml \
    --profile=cubi-v1 \
    --restart-times 2 \
    --keep-going \
    --rerun-incomplete \
    --verbose \
    --use-conda \
    --conda-prefix=/data/gpfs-1/users/cofu10_c/scratch/P3473/envs/ 
    #--touch \
    #--skip-script-cleanup \
    #--reason 
    #--until annovar
    #--until table_to_anno \
