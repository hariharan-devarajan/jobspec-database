#!/bin/bash
# LSF scheduler options
#BSUB -P CSC190SPECFEM
#BSUB -J go_benchmark 
#BSUB -o OUTPUT_FILES/job.o

###################################################
## USER PARAMETERS
## summitdev: compute nodes have 4 GPU cards (P100) and 2 x 10-core (Power8) CPUs
## summit:    compute nodes have 6 GPU cards (V100) and 2 x 21-core (Power9, 4 hardware threads per core by default) CPUs

# 30min walltime
#BSUB -W 00:30

#use: bsub -env "all,JOB_FEATURE='gpudefault,gpumps'" < job_mesher_solver.bash
#
# summitdev:
# see: https://www.olcf.ornl.gov/kb_articles/summitdev-quickstart/
# The new LSF -nnodes #nodes flag replaces the previous -n #slots flag.
# Note: -nnodes requests nodes while -n requests slots/cores.
# All batch scripts should be updated to use #BSUB -nnodes #nodes
#
# summit:
# https://www.olcf.ornl.gov/for-users/system-user-guides/summit/running-jobs/
#
#BSUB -nnodes 16

# jsrun options:
#--nrs		-n	Number of resource sets
#--tasks_per_rs	-a	Number of MPI tasks (ranks) per resource set
#--cpu_per_rs	-c	Number of CPUs (cores) per resource set.
#--gpu_per_rs	-g	Number of GPUs per resource set
#--bind		-b	Binding of tasks within a resource set. Can be none, rs, or packed:#
#--rs_per_host	-r	Number of resource sets per host
#--latency_priority	-l	Latency Priority. Controls layout priorities. Can currently be cpu-cpu or gpu-cpu
#--launch_distribution	-d	How tasks are started on resource sets
#
# example: jsrun -n1 -g1 -a1 -c1 

# resource sets (a node contains 42 slots)
##BSUB -n 4

# features
#
##BSUB -alloc_flags gpumps  # gpu multiple MPI process per GPU
##BSUB -alloc_flags smt2    # hyperthreading 2 threads per single core


NPROC=96

###################################################

###export COMPUTE_PROFILE=1
###export COMPUTE_PROFILE_LOG=specfem3d_profile.log

# in case multiple mpi processes share same gpu
#export CRAY_CUDA_PROXY=1
#export CUDA_MPS_CLIENT=1

echo "running simulation: `date`"
echo "directory: `pwd`"
echo

mkdir -p OUTPUT_FILES
mkdir -p DATABASES_MPI

# cleanup
rm -rf OUTPUT_FILES/*
rm -rf DATABASES_MPI/*

# obtain job information
echo "nodes: $LSB_HOSTS"
echo
echo "job ID: $LSB_JOBID"
echo

# runs mesher
echo
echo "running mesher..."
echo `date`
jsrun -n96 -r6 -g1 -a1 -c1 ./bin/xmeshfem3D
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
jsrun -n96 -r6 -g1 -a1 -c1 ./bin/xspecfem3D
if [[ $? -ne 0 ]]; then exit 1; fi

# cleanup
rm -rf DATABASES_MPI

echo
echo "see results in directory: OUTPUT_FILES/"
echo
echo "done: `date`"


