#!/bin/bash

#BSUB -W 03:59                # How much time does your job need (HH:MM)
#BSUB -q short                # Which queue
#BSUB -J "hpccRunFlip[1-10]"  # Job Name (and array size)
#BSUB -R rusage[mem=1000]     # Memory requirements (in MB)
#BSUB -n 1                    # Number of nodes to use
#BSUB -o "./%J.o"             # Specifies name of the output file
#BSUB -e "./%J.e"             # Specifies name of the error file
#BSUB -w 'done(hpccRunPre)'   # wait to submit until down with runPre job

#cd coinFlip/                  # change directories to groundfish-MSE
module load R/3.4.0           # load R module

Rscript ./runFlip.R --vanilla       # Run the runFlip.R code

echo "runFlip complete"       # Print statement indicating job is done