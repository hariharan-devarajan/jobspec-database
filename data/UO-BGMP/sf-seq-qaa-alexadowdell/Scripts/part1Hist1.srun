#!/bin/bash

#SBATCH --partition=short        ### Indicate want long node
#SBATCH --job-name=AD_R1     ### Job Name
#SBATCH --output=AD_R1.out         ### File in which to store job output
#SBATCH --error=AD_R1.err          ### File in which to store job error messages
#SBATCH --time=0-12:00:00	### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job
#SBATCH --ntasks-per-node=28     ### Nuber of tasks to be launched per Node

ml easybuild GCC/6.3.0-2.27 OpenMPI/2.0.2 Python/3.6.1

python part1Hist1.py -f 22_3H_both_S16_L008_R1_001.fastq 
