#!/bin/sh
#BSUB -J P3
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
# EXE="lmp_plumed"
totalCores=12

P0=3.15 ## unit nm

# for AT in 1.2  1.0  0.8  0.6   
# for AT in 0.5  0.4  0.3  0.2  0.1  0.0
# for AT in -0.1 -0.2 -0.3 -0.4 -0.5 -0.6
# for AT in -0.6 -0.8 -1.0 -1.2
for AT in -1.00 -0.90 -0.80 -0.70 -0.60 -0.50 -0.40 -0.30 -0.20
# for AT in -0.10  0.00  0.10  0.20  0.30  0.40  0.50  0.60  0.70  0.80
# for AT in  0.90  1.00  1.10  1.20  1.30  1.40  1.50  1.60  1.70  1.80
# for AT in  1.90  2.00  2.10  2.20  2.30  2.40  2.50 
do
cd sample_${AT}
bsub < tsjobs 

cd ..
done
