#!/bin/sh
#BSUB -J K.T
#BSUB -q priority
#BSUB -n 24
#BSUB -o %J.opt -e %J.err
#BSUB -W 1000:00
#BSUB -a intelmpi
#BSUB -R "span[ptile=12]"

#mpirun.lsf -np 12 /home/xuzp/bin/vasp_isif  >& log
#mpirun.lsf -np 24 /home/xuzp/bin/vasp5.3.5_isif_cc  >& log
mpirun.lsf -np 24 /home/xuzp/bin/vasp.5.4.4.std  >& log
