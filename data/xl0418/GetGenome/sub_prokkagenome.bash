#!/bin/bash
## now loop through the above array

#SBATCH --time=00:10:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -J "Prokka"   # job name
#SBATCH --mail-user=liangxu@caltech.edu   # email address

module load parallel/20180222
module load singularity/3.3.0

sh prokka_genomes.sh 2 221117-1910.P16N-S.16S.dna-sequences.tsv


