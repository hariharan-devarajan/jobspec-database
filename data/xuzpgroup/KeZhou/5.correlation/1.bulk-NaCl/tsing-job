#!/bin/sh
#BSUB -J bulk-15
#BSUB -q priority
#BSUB -n 12
#BSUB -o %J.output -e %J.err
#BSUB -W 240:00
#BSUB -a intelmpi
#BSUB -R "span[ptile=12]"

mpirun.lsf -np 12 /home/xuzp/bin/lmp_zhou <eq.GO.ions.in >& log

