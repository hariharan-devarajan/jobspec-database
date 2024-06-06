#!/bin/sh
#BSUB -J E=1.0.K
#BSUB -q priority
#BSUB -n 24
#BSUB -o %J.output -e %J.err
#BSUB -W 240:00
#BSUB -a intelmpi
#BSUB -R "span[ptile=12]"

#mpirun.lsf -np 24 /home/xuzp/bin/lmp_zhou <eq.GO.ions.in >& log
mpirun.lsf -np 24 /home/xuzp/bin/lmp_zhou.11.Aug.2017 <eq.GO.ions.in >& log

