#!/bin/sh
#BSUB -J GOKXX
#BSUB -q priority
#BSUB -n 24
#BSUB -o %J.output -e %J.err
#BSUB -W 240:00
#BSUB -a intelmpi
#BSUB -R "span[ptile=12]"

mpirun.lsf -np 24 /home/xuzp/bin/lmp_zhou <eq.GO.ions.in >& log

