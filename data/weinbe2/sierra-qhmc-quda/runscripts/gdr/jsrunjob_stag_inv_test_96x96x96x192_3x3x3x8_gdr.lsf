#!/bin/bash

#BSUB -nnodes 54
# #BSUB -q pdebug
# #BSUB -G guests
#BSUB -q pbatch
#BSUB -G callat_g
#BSUB -W 45
#BSUB -o stag_profile.96x96x96x192.n54.3x3x3x8.gdr.%J
## Uncomment to make this interactive
# #BSUB -Is
##
#BSUB -core_isolation 2
####BSUB -x

#export HDF5_DISABLE_VERSION_CHECK=2

#source env.sh
module load gcc/7.3.1
module load cmake/3.12.1

#export LD_LIBRARY_PATH=/usr/workspace/wsb/gambhir1/software/sierra/install/sierra/hdf5/lib:$LD_LIBRARY_PATH

#export OMP_NUM_THREADS=1

#cd /p/gscratch1/gambhir1/runs/xmls/props

echo "About to change dir"

#cd /usr/workspace/wsb/weinberg/runs/
export QUDA_RESOURCE_PATH=$(pwd)/tunecache_96x96x96x192_3x3x3x8_gdr/
export QUDA_ENABLE_GDR=1 # no GDR... for now

echo "Changed dir"

#Recursively submit jobs.
#if [ $c -lt $max_c ]
# then
#  bsub < jsrunjob_solve_prop.lsf
#fi

APP="/nfs/tmp2/lsd/weinberg/new_bin/scidac/quda/build/tests/staggered_invert_test"
ARG="--rank-order col --dim 32 32 32 24 --gridsize 3 3 3 8 --load-gauge l96t192m00125b480_config.1894.lime --compute-fat-long true --dslash-type staggered --test 1 --niter 1000000 --tol 1e-10 --prec double --prec-sloppy half --mass 0.00125"
PROG="$APP $ARG"
#APRUNVAR="jsrun -p16 -r4 --bind=none ./sierra_jsrun.reorder.sh $PROG"
APRUNVAR="jsrun -p216 -g1 -M \"-gpu\" --bind=none ./sierra_jsrun.reorder.sh $PROG"
#$APRUNVAR -i $base.ini.xml -o $base.out > $base.stdout 2>&1 

echo "About to run job"
echo "${APRUNVAR}"

$APRUNVAR 

echo "Job ran."

