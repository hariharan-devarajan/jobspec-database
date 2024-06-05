#!/bin/bash 

# job name  
#SBATCH --job-name=lc1_job
# job output
#SBATCH --output=lc1_job.out
# errors
#SBATCH --error=lc1_job.err
#time needed (hh:mm)
#SBATCH --time=15
#number of nodes you are requesting
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
# memory per node
#SBATCH --mem=1G
# set up email
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=logan.b.smith@vanderbilt.edu

# task requested
module load GCC Python
module load Pysam
python livecoding.py -b 2.bam
