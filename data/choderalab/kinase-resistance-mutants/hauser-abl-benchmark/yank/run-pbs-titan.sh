#!/bin/bash
#    Begin PBS directives
#
# Account to charge: our project number
#PBS -A chm126
#
# Set job name
#PBS -N resistance
#
# Capture output and error
#PBS -j oe
#
# Production limit is six hours, but jobs can be chained
#PBS -l walltime=6:00:00,nodes=352
#
# Use atlas scratch storage
#PBS -l gres=atlas1%atlas2
#
# Start GPUs in shared mode for YANK to work
#PBS -l feature=gpudefault
#
#    End PBS directives and begin shell commands

# MODIFY ME: Change HOME to point to working directory on atlas scratch
export HOME=/lustre/atlas/scratch/jchodera1/chm126

# Set up miniconda; LD_LIBRARY_PATH must be manually set because of cray weirdness when installing conda
export MINICONDA="$HOME/miniconda3"
export PATH="$MINICONDA/bin:$PATH"
export LD_LIBRARY_PATH=$MINICONDA/lib:$LD_LIBRARY_PATH

# MODIFY ME: Change directory to working directory on atlas scratch
cd $HOME/kinase-resistance-mutants/hauser-abl-benchmark/yank/

# Configure CUDA
module load cudatoolkit
export OPENMM_CUDA_COMPILER=`which nvcc`
echo $OPENMM_CUDA_COMPILER

# Set up mpi environment
module remove PrgEnv-pgi
module add PrgEnv-gnu
module add cray-mpich

# Set the OpenEye license, but not all the openeye tools work here
export OE_LICENSE="/lustre/atlas/scratch/jchodera1/chm126/.openeye/oe_license.txt"

# Run YANK, one MPI process per node
aprun -n $PBS_NUM_NODES -N 1 -d 16 yank script --yaml=allmuts-sams.yaml
