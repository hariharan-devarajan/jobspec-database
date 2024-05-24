#!/bin/bash -l 
#SBATCH --time=96:00:00
#SBATCH --ntasks=20
#SBATCH --tmp=20g
#SBATCH --mem=20g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jaber038@umn.edu
#SBATCH -p small,large,amdlarge,amdsmall 

source /etc/profile.d/modules.sh 
module load impi 
module load R/4.1.0 

conda activate ruv
make all 

