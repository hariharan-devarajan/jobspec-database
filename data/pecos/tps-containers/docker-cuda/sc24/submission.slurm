#!/bin/bash

#SBATCH -J tps_bte           # Job name
#SBATCH -o tps_bte.o%j       # Name of stdout output file
#SBATCH -e tps_bte.e%j       # Name of stderr error file
#SBATCH -p gpu-a100                  # Queue (partition) name
#SBATCH -N 2                       # Total # of nodes (must be 1 for serial)
#SBATCH -n 6                       # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 00:10:00                # Run time (hh:mm:ss)
#SBATCH --mail-type=all            # Send email at begin and end of job
#SBATCH -A FTA-SUB-Ghattas               # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=uvilla@oden.utexas.edu

# Any other commands must follow all #SBATCH directives...
module purge
module load gcc/11.2.0 mvapich2/2.3.7 tacc-apptainer/1.1.8 cuda/12.2
ml list



MV2_SMP_USE_CMA=0 ibrun apptainer run --nv tps-bte-ls6_latest.sif /tps/build-gpu/src/tps-bte_0d3v.py -run input.ini