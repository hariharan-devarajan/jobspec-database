#!/bin/sh
#BSUB -J PLUMED+US
#BSUB -q priority
#BSUB -n 12
#BSUB -o %J.output -e %J.err
#BSUB -W 240:00
#BSUB -a intelmpi
#BSUB -R "span[ptile=12]"


############################################################################
# Definition of variables
############################################################################
EXE="/home/xuzp/bin/lmp_zhou.plumed"
EXE="lmp_plumed"
totalCores=4


#sh run.sh
mpirun -np ${totalCores} ${EXE} < eq.GO.ions.in  # > out.lmp
# mpirun.lsf -np ${totalCores} ${EXE} < eq.GO.ions.in >& log


