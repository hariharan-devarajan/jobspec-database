#!/bin/bash
# LSF scheduler options
#BSUB -P CSC190SPECFEM 
#BSUB -J SPECFEM3D_mesher_solver
#BSUB -o OUTPUT_FILES/job_mesher_solver.o
#
###################################################
#
## USER PARAMETERS
## summitdev: compute nodes have 4 GPU cards (P100) and 2 x 10-core (Power8) CPUs
#
# 30min walltime
#BSUB -W 00:30
# compute slots (a node contains 20 slots)
##BSUB -n 1
#
#use: bsub -env "all,JOB_FEATURE='gpudefault,gpumps'" < job_mesher_solver.bash
#
#
# see: https://www.olcf.ornl.gov/kb_articles/summitdev-quickstart/
# The new LSF -nnodes #nodes flag replaces the previous -n #slots flag.
# Note: -nnodes requests nodes while -n requests slots/cores.
# All batch scripts should be updated to use #BSUB -nnodes #nodes
#BSUB -nnodes 1


NPROC=1

###################################################

###export COMPUTE_PROFILE=1
###export COMPUTE_PROFILE_LOG=specfem3d_profile.log

# in case multiple mpi processes share same gpu
#export CRAY_CUDA_PROXY=1
#export CUDA_MPS_CLIENT=1

echo "running simulation: `date`"
echo "directory: `pwd`"
echo

# obtain job information
echo "nodes: $LSB_HOSTS"
echo
echo "job ID: $LSB_JOBID"
echo

# runs mesher
echo
echo "running mesher..."
echo `date`
jsrun -n$NPROC -g1 -a1 -c1  ./bin/xmeshfem3D
if [[ $? -ne 0 ]]; then exit 1; fi

cp -p OUTPUT_FILES/addr*.txt DATABASES_MPI/
#cp -p DATABASES_MPI/addr*.txt OUTPUT_FILES/

# stores setup
cp DATA/Par_file OUTPUT_FILES/
cp DATA/CMTSOLUTION OUTPUT_FILES/
cp DATA/STATIONS OUTPUT_FILES/

# runs simulation
echo
echo "running solver..."
echo `date`
jsrun -n$NPROC -g1 -a1 -c1 ./bin/xspecfem3D
if [[ $? -ne 0 ]]; then exit 1; fi

echo
echo "see results in directory: OUTPUT_FILES/"
echo
echo "done: `date`"


