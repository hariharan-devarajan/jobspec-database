#!/bin/bash
#
#PBS -N ZIF8_1-1-2_H2O_1-1-05
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=9

source $VSC_DATA_VO/vsc40685_apps/activate.sh
module load LAMMPS/3Mar2020-intel-2019b-Python-3.7.4-kokkos

ORIGDIR=$PBS_O_WORKDIR
WORKDIR=/local/$PBS_JOBID
SCRIPTDIR1=$VSC_SCRATCH_VO/vsc40686/NPT_simulations/ZIF-8/waterIntrusion/water_gradient/scripts
SCRIPTDIR2=$VSC_SCRATCH_VO/vsc40686/NPT_simulations/ZIF-8/waterIntrusion/water_gradient/zif8_1-1-2_H2O_1-1-05/scripts
mkdir -p $WORKDIR

cp $SCRIPTDIR2/init.chk $WORKDIR/.
cp $SCRIPTDIR1/pars.txt $WORKDIR/.
cp $SCRIPTDIR1/ghostatoms.py $WORKDIR/.
cp $SCRIPTDIR1/rcut_12.0.table $WORKDIR/.
cp $ORIGDIR/ymd.py $WORKDIR/.

cd $WORKDIR
mpirun -np 9 python ymd.py

cp $WORKDIR/* $ORIGDIR/.

cd $ORIGDIR
rm -rf $WORKDIR

