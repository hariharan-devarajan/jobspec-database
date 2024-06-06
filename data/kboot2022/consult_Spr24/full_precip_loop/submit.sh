#!/bin/bash

#BSUB -n 1 		## request nodes
#BSUB -W 1440 		## work time minutes
##BSUB -R "rusage[mem=192GB]" ## set memory for job
#BSUB -J full_loop 	## run name
#BSUB -o out.%J 	## output
#BSUB -e err.%J 	## error
##BSUB -x ##Use exclusive only if necessary, uncomment if job spawns additional threads
module load matlab 
matlab -nodisplay -nosplash -nodesktop -singleCompThread -r "run('Precip_loop_Hazel_mean.m');exit;"
