#!/bin/bash
#
#BSUB –J MPIJob ### set the job Name
#BSUB -q short ### specify queue
#BSUB -n 40 ### ask for number of cores (default: 1)
#BSUB –R “span[ptile=40]” ### ask for 40 cores per node
#BSUB -W 10:00 ### set walltime limit: hh:mm
#BSUB -o /result/output/stdout_yue_%J.out ### Specify the output and error file. %J is the job-id
#BSUB -e /result/error/stderr_yue_%J.err ### -o and -e mean append, -oo and -eo mean overwrite
# here follow the commands you want to execute
# load the necessary modules
# NOTE: this is just an example, check with the available modules
#module load cuda/10.0
#module load mpi/openmpi/3.1.2_gcc
python3 /work/cse-liuy/Neat/newevolve.py  0 500 0 0
