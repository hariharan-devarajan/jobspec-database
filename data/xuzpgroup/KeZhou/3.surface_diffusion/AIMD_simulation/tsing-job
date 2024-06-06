#!/bin/sh
#BSUB -J 4L
#BSUB -q priority
#BSUB -n 36
#BSUB -o %J.output -e %J.err
#BSUB -W 240:00
#BSUB -a intelmpi
#BSUB -R "span[ptile=12]"

#mpirun.lsf -np 60 /home/xuzp/bin/cp2k2.4.popt -i cl.inp -o log
#mpirun.lsf -np 60 /home/xuzp/bin/cp2k2.4.popt -i relax.inp -o log
#mpirun.lsf -np 60 /apps/soft/cp2k/cp2k2.5/cp2k-2.5.1/exe/Linux-x86-64-intel/cp2k.popt -i relax.in -o log
mpirun.lsf -np 36 /apps/soft/cp2k/cp2k2.5/cp2k-2.5.1/exe/Linux-x86-64-intel/cp2k.popt -i cl.inp -o log

