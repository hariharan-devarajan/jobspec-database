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

P0=3.14925 ## unit nm


# for AT in -1.00 -0.90 -0.80 -0.70 -0.60 -0.50 -0.40 -0.30 -0.20
for AT in -0.10  0.00  0.10  0.20  0.30  0.40  0.50  0.60  0.70  0.80
# for AT in  0.90  1.00  1.10  1.20  1.30  1.40  1.50  1.60  1.70  1.80
# for AT in  1.90  2.00  2.10  2.20  2.30  2.40  2.50 
do
mkdir sample_${AT}
cp FF.water.ion.prm H10.data tsjobs sample_${AT}
# dest=`echo "scale=3;$AT*10.0"|bc`
dd=`echo "scale=4;(${P0}+$AT)*10.0"|bc`
sed -e 's/XXX/'${dd}'/g' ./inp.temp > ./sample_${AT}/eq.GO.ions.in

cat >./sample_${AT}/plumed.dat << EOF
c1: CENTER ATOMS=1221,1771
#d1:  DISTANCE ATOMS=c1,1855
d1:  DISTANCE ATOMS=c1,5871 COMPONENTS

restraint-d1: RESTRAINT ARG=d1.x KAPPA=1000.0 AT=${AT}
PRINT STRIDE=250 ARG=d1.x,restraint-d1.bias FILE=COLVAR.${AT}

EOF
cd sample_${AT}
#sh run.sh
# mpirun -np ${totalCores} ${EXE} < eq.GO.ions.in  # > out.lmp
mpirun.lsf -np ${totalCores} ${EXE} < eq.GO.ions.in >& log
#mpirun -np 4 lmp_plumed  < eq.GO.ions.in
#mv all.ele.dat  all.ele.${AT}.dat 
cd ..
done
