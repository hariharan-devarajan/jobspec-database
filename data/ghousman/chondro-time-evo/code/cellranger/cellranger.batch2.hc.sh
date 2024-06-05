#!/bin/bash

# sbatch submission script to run main snakemake process
# submits individual jobs from the compute node

#SBATCH --job-name=cellranger
#SBATCH --output=cellranger.batch2.hc.out
#SBATCH --error=cellranger.batch2.hc.err
#SBATCH --account=pi-gilad
#SBATCH --time=36:00:00
#SBATCH --partition=caslake
#SBATCH --mem=128G
#SBATCH --tasks-per-node=12

/project2/gilad/ghousman/cellranger/cellranger-7.0.0/bin/cellranger multi --id human_chimp_chondro_time_batch2_hc \
                                                                          --csv ./../chondro-time-evo/code/cellranger/cellranger.batch2.hc.csv \
                                                                          --localcores 12 \
                                                                          --localmem 48
