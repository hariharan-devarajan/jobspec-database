#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --qos=preemptable
#SBATCH --account=blanca-lundquist
#SBATCH --output=log.fe.%j
#SBATCH --export=NONE
#SBATCH --mail-type=ALL
#SBATCH --mail-user="misa5952@colorado.edu"
#SBATCH --job-name=test_FE
#SBATCH --gres=gpu

source /projects/misa5952/FastEddy/FastEddy_model/SRC/FEMAIN/setBeforeCompiling
# export SLURM_EXPORT_ENV=ALL
# export I_MPI_FABRICS=shm:ofa
export I_MPI_FALLBACK=1
export I_MPI_SHM_LMT=shm

module list
nvidia-smi > hhh
hostname

ulimit -s unlimited

FE=$PWD/SRC/FEMAIN/FastEddy

mpirun -np $SLURM_NTASKS $FE NBL_params.in

# $MPIRUN -np $SLURM_NTASKS $WRF_ROOT/wrf.exe

exit 0
