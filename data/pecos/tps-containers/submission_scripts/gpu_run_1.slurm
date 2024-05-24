#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Lonestar6 AMD Milan nodes
#
#   *** Parallel Job in GPU queue***
# 
#----------------------------------------------------

#SBATCH -J tps_cylinder           # Job name
#SBATCH -o tps_cylinder.o%j       # Name of stdout output file
#SBATCH -e tps_cylinder.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-small                # Queue (partition) name
#SBATCH -N 1                       # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                       # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:10:00                # Run time (hh:mm:ss)
#SBATCH --mail-type=all            # Send email at begin and end of job
#SBATCH -A FTA-SUB-Ghattas               # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=uvilla@oden.utexas.edu

# Any other commands must follow all #SBATCH directives...
module purge
module load gcc/11.2.0 mvapich2/2.3.7 tacc-apptainer/1.1.8 cuda/11.4
ml list

export ORG=uvilla
export IMAGE_NAME=tps_env_parla
export TAG=gnu9-mvapich2-sm_80


export IMAGE=$(pwd)/${IMAGE_NAME}_$TAG.sif

rm -f $IMAGE
apptainer pull docker://$ORG/$IMAGE_NAME:$TAG

export EXE=$(pwd)/tps/build-gpu/src/tps
export PYEXE=$(pwd)/tps/build-gpu/src/tps-time-loop.py
export RAW_EXE=$(pwd)/tps/build-gpu/src/.libs/tps
export RAW_LIB=$(pwd)/tps/build-gpu/src/.libs/libtps.so

echo "Apptainer nvcc"
ibrun -n 1 apptainer exec --nv $IMAGE which nvcc
echo "NVIDIA SMI"
ibrun -n 1 apptainer exec --nv $IMAGE nvidia-smi
echo "ldd $RAW_EXE"
ibrun -n 1 apptainer exec --nv $IMAGE ldd $RAW_EXE
echo "ldd $RAW_LIB"
ibrun -n 1 apptainer exec --nv $IMAGE ldd $RAW_LIB

echo "----"
ibrun -n 1 apptainer exec --nv $IMAGE ldd $RAW_EXE | grep not
ibrun -n 1 apptainer exec --nv $IMAGE ldd $RAW_LIB | grep not
echo "----"

cd tps/build-gpu/test
echo "Test $EXE"
MV2_SMP_USE_CMA=0 ibrun apptainer exec --nv $IMAGE $EXE -run inputs/coupled-3d-boltzmann.dtconst.ini  
echo "Test $PYEXE"
MV2_SMP_USE_CMA=0 ibrun apptainer exec --nv $IMAGE $PYEXE -run inputs/coupled-3d-boltzmann.dtconst.ini 
