#!/bin/bash
#PBS -l mem=126gb,nodes=1:ppn=16,walltime=96:00:00
#PBS -m abe
#PBS -j oe
#PBS -M rmoran@umn.edu
#PBS -q sb128

# Load modules
module load zlib/1.2.8
module load xz-utils/5.2.2_intel2015update3
module load parallel/20160822
module load bwa/0.7.4
module load samtools/1.7 


# Specify the job command list to use (cave or surface)
#CMD_LIST="/home/mcgaughs/shared/Datasets/Reads_ready_to_align/raw_bwa_commands_surface.txt"
CMD_LIST="/home/mcgaughs/shared/Datasets/Reads_ready_to_align/raw_bwa_commands_cavefish.txt"

CMD="$(sed "${PBS_ARRAYID}q;d" ${CMD_LIST})"
eval ${CMD}
