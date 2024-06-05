#!/bin/bash
#SBATCH --job-name=abcd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu 20000
#SBATCH -p lab_fat_c
#SBATCH -o 
#SBATCH -e 

module purge
module load MATLAB/R2018b
test
matlab -nodesktop -nosplash -r "sub_n=$1;Step_one_site16_code_par.m;quit"
