#! /bin/bash -l
 
#SBATCH --partition=panda   # cluster-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=sm_read_simulation
#SBATCH --time=16:00:00   # HH/MM/SS
#SBATCH --mem=15G   # memory requested, units available: K,M,G,T
#SBATCH --output new_10S10C1_readsim.out
#SBATCH --error new_10S10C1_readsim.err

source ~/.bashrc
snakemake --cores 1
