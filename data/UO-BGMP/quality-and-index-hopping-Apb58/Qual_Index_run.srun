#!/bin/bash
#SBATCH --partition=short       ### Partition (like a queue in PBS)
#SBATCH --job-name=ABQIH        ### Job Name
#SBATCH --output=qual.out      ### File in which to store job output
#SBATCH --error=qual.err       ### File in which to store job error messages
#SBATCH --time=0-05:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job
#SBATCH --ntasks-per-node=28    ### Number of tasks to be launched per Node

module load easybuild
module load prl
module load python/3.6.0
cd /home/abubie/qual_ind_swp
./Qual_Index_Swp.py -r1 /projects/bgmp/2017_sequencing/1294_S1_L008_R1_001.fastq -r2 /projects/bgmp/2017_sequencing/1294_S1_L008_R4_001.fastq -i1 /projects/bgmp/2017_sequencing/1294_S1_L008_R2_001.fastq -i2 /projects/bgmp/2017_sequencing/1294_S1_L008_R3_001.fastq -indf /home/abubie/qual_ind_swp/index.tsv -q_cut 35

echo $"Index/Qual analysis is complete"
