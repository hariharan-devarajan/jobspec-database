#!/bin/bash
#SBATCH -p astyanax,small,amdsmall,cavefish
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120gb
#SBATCH -t 12:00:00
#SBATCH -J surf.discoal
#SBATCH --array=1-23

cd /home/mcgaughs/shared/Software/diploSHIC
discoal="/home/mcgaughs/shared/Software/discoal/discoal"


CMD_LIST="Surface_discoal_commands_w_3popsizes_2.txt"
#CMD_LIST="discoal_commands_orig_params_w_popsize.txt"
CMD="$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${CMD_LIST})"
eval ${CMD}
