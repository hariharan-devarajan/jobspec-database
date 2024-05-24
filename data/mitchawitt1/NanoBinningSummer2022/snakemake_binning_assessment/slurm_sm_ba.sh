#! /bin/bash -l
 
#SBATCH --partition=panda   # cluster-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=sm_CATBAT
#SBATCH --time=16:00:00   # HH/MM/SS
#SBATCH --mem=12G   # memory requested, units available: K,M,G,T
#SBATCH --output sm_catbat.out
#SBATCH --error sm_catbat.err

source ~/.bashrc
cd /athena/ihlab/scratch/miw4007/simulation_test/tools/snakemake_binning_assessment/
snakemake --cores 4
exit
