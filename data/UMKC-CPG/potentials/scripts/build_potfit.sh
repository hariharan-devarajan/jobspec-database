#/usr/bin/env bash

set -eux

BUILD_DIR="_build"
POTFIT_REPO="https://github.com/potfit/potfit.git"
POTFIT_RELEASE="0.7.1"

######################################################################
# User configuration
POT_OPTION="potfit_mpi_tersoff_apot"
JOB_DIR="${HOME}/data/alpha20/25potfitaboron"
CONFIG_FILE="paramfile"
######################################################################


######################################################################
# Compilation step.
if [ -d ${BUILD_DIR} ]; then
    rm -rf ${BUILD_DIR}
fi

git clone ${POTFIT_REPO} ${BUILD_DIR}
cd ${BUILD_DIR}
git checkout tags/${POTFIT_RELEASE}

sed -i.bak -e 's/mpicc/mpiicc/' -e 's/^BIN_DIR/#&/' Makefile

module load intel

unset BIN_DIR

make ${POT_OPTION}
cp ${POT_OPTION} ${JOB_DIR}
cd ${JOB_DIR}


######################################################################
# Slurm parameters 

#SBATCH -p Lewis
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4  # used for MPI codes, otherwise leave at '1'
#DISABLED #SBATCH --ntasks-per-node=4  # don't trust SLURM to divide the cores evenly when you use more than one node
#SBATCH --cpus-per-task=1  # cores per task; set to one if using MPI
#SBATCH -J  aboron
#SBATCH -o  aboron.o%J
#SBATCH -e  aboron.e%J

##DISABLED
#SBATCH --mem-per-cpu=2G
######################################################################


######################################################################
# Start MPI job
module load mpich/mpich-3.2-intel

# MPI flag for explicit saftey
export PSM_RANKS_PER_CONTEXT=2

mpirun -np ${SLURM_NTASKS} ./${POT_OPTION} ${CONFIG_FILE} | tee output

wait
######################################################################
