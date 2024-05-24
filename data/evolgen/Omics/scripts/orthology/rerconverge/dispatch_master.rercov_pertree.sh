#!/usr/bin/bash

#SBATCH --job-name=rer_masterfilename
#SBATCH --account=co_genomicdata
#SBATCH --partition=savio22_bigmem
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH -o /global/scratch2/rohitkolora/Rockfish/Genomes/orthologs/Lifted/Sebastes_55/rerconverge/list_configs/masterfilename/log
#SBATCH -e /global/scratch2/rohitkolora/Rockfish/Genomes/orthologs/Lifted/Sebastes_55/rerconverge/list_configs/masterfilename/error

module load gcc/4.8.5 openmpi # or module load intel openmpi, ALWAYS required
#ht_helper.sh -m "samtools" -t taskfile

#snakemake -s multi_rerconv_trees_Snakefile_new --configfile /global/scratch2/rohitkolora/Rockfish/Genomes/orthologs/Lifted/Sebastes_55/rerconverge/list_configs/masterfilename/config_masterfilename.json -j 25 -npr

cd /global/scratch2/rohitkolora/Rockfish/Genomes/orthologs/Lifted/Sebastes_55/rerconverge/list_configs/masterfilename/ ;
rm -fr .snakemake ;
snakemake -s /global/scratch2/rohitkolora/Rockfish/Genomes/orthologs/Lifted/Sebastes_55/rerconverge/multi_rerconv_trees_Snakefile_new --configfile /global/scratch2/rohitkolora/Rockfish/Genomes/orthologs/Lifted/Sebastes_55/rerconverge/list_configs/masterfilename/config_masterfilename.json -j 7 -k


