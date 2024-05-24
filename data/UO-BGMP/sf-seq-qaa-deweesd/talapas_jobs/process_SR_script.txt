#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=PS_trimmed_data
#SBATCH --output=Process_S2.out
#SBATCH --error=Process_S2.err
#SBATCH --time=0-03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mail-user=daned@uoregon.edu
#SBATCH --mail-type=all

module purge

module load slurm easybuild intel/2017a Stacks/1.46

process_shortreads -P -i fastq -1 /home/daned/bi624/ps1/8_2F_fox_S7_L008_R1_001.fastq 
-2 /home/daned/bi624/ps1/8_2F_fox_S7_L008_R2_001.fastq 
-o /home/daned/bi624/ps1/trimmed_data 
--adapter_1 AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC --adapter_2 AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT --adapter_mm 2



process_shortreads -P -i fastq -1 /home/daned/bi624/ps1/31_4F_fox_S22_L008_R1_001.fastq -2 /home/daned/bi624/ps1/31_4F_fox_S22_L008_R2_001.fastq 
-o /home/daned/bi624/ps1/output 
--adapter_1 AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC --adapter_2 AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT --adapter_mm 2
