#!/bin/bash

#SBATCH --partition=batch
#SBATCH --job-name=Zhang_Nextflow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --mem=8gb

#Replace this with your UGA email to get notified on completion
#SBATCH --mail-user="bjl34716@uga.edu"
#SBATCH --mail-type=BEGIN,END,FAIL

SUBDIR=$(pwd)

if [[ ! -d /scratch/bjl34716/nf_dev/gg-catalog ]]; then
    mkdir -p /scratch/bjl34716/nf_dev/gg-catalog
fi

cd /scratch/bjl34716/nf_dev/gg-catalog

module load Nextflow/22.04.5
nextflow run lorentzben/gg-catalog-nf \
        -r main \
        -with-tower \
        -c /home/bjl34716/applegate/villegas/compare_methods/.nextflow/config/gacrc.config \
        -profile slurm,singularity \
        -params-file /home/bjl34716/my_utils/gg-catalog/code/params/concat_params.yaml \
        -latest \
        -resume
        

