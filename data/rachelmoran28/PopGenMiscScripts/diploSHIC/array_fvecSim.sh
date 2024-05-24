#!/bin/bash
#SBATCH -p small,amdsmall,astyanax
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60gb
#SBATCH -t 3:00:00
#SBATCH -J fvecSim
#SBATCH --array=1-23

#generating feature vector stats
cd /home/mcgaughs/shared/Software/diploSHIC

CMD_LIST="Surface.fvecSim.commnads.txt"                                                                                                                                                               
#CMD_LIST="Pachon.fvecSim.commnads.txt"
#CMD_LIST="Pachon.fvecSim.commnads.test.txt"
#CMD_LIST="fvecSim.commnads.retry.txt"

CMD="$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${CMD_LIST})"
eval ${CMD}
