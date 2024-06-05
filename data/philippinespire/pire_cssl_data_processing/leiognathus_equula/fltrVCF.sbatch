#!/bin/bash -l

#SBATCH --job-name=fltrVCF
#SBATCH -o fltrVCF-%j.out
#SBATCH -p main
#SBATCH -c 40
##SBATCH --time=96:00:00

#to run use the following command:
#sbatch fltrVCF.sbatch config.fltr.ind

enable_lmod
module load container_env ddocent/2.7.8
export SINGULARITY_BIND=/home/e1garcia
export PARALLEL_SHELL=/bin/bash

#module load container_env ddocent
#module load container_env perl/5.22.0
#module load container_env vcftools/0.1.15
#module load container_env rad_haplotyper/1.1.5
#module load container_env parallel
#module load container_env vcflib/1.0
#module load container_env samtools
#module load container_env R/gcc7/3.5.3

bash /home/e1garcia/shotgun_PIRE/pire_cssl_data_processing/leiognathus_equula/fltrVCF.bash -s $1
