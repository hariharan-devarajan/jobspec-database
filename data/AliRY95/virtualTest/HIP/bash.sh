#!/bin/bash
#SBATCH --account=cad14948
#SBATCH --job-name=VirtualFunction
#SBATCH --constraint=MI250
#SBATCH --nodes=1
#SBATCH --exclusive
##SBATCH --reservation=hackathon
#SBATCH --time=00:01:00
#SBATCH --gpus-per-node=1
#SBATCH --output=out.out
#SBATCH --error=err.err

######################################################################
#                           ULIMIT                                   #
######################################################################
# FOR HIRLAM
ulimit -s unlimited
ulimit -S -s unlimited  # stack
ulimit -S -d unlimited  # data area
ulimit -S -c unlimited
ulimit -S -m unlimited  # memory
ulimit -S -n unlimited  # memory
ulimit -S -q unlimited  # memory
ulimit -l unlimited

module purge

module load PrgEnv-amd
module load CPE-23.02-rocmcc-5.3.0-GPU-softs
module list

rocminfo
module list

export HIP_PATH=/opt/rocm-5.5.1/hip

cd ${SLURM_SUBMIT_DIR}
make clean
make
run -n 1 -- ./HIP_virtualTest
