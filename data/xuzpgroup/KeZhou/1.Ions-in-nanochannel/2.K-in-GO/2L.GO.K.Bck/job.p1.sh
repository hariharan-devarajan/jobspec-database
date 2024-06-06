#!/bin/sh
#BSUB -J K2GO
#BSUB -q priority
#BSUB -n 24
#BSUB -o %J.output -e %J.err
#BSUB -W 240:00
#BSUB -a intelmpi
#BSUB -R "span[ptile=12]"

###
for i in 10 20 30 40 50 60 70 80 90 
do 
cd rerun.GO.bck.${i}
mpirun.lsf -np 24 /home/xuzp/bin/lmp_zhou <eq.GO.ions.in >& log
cd ..
done 
