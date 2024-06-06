#!/bin/sh
#BSUB -J SMD.6.8.Na
#BSUB -q priority
#BSUB -n 12
#BSUB -o %J.output -e %J.err
#BSUB -W 240:00
#BSUB -a intelmpi
#BSUB -R "span[ptile=12]"


############################################################################
# Definition of variables
############################################################################
EXE="/home/xuzp/bin/lmp_zhou"

# EXE="lmp_plumed"
totalCores=12

mpirun.lsf -np ${totalCores} ${EXE} < SMD.inp >& log

