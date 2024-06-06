#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J weak
#Output and error (also --output, --error):
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#Initial working directory (also --chdir):
#SBATCH -D ./
#Notification and type
##SBATCH --mail-type=END
##SBATCH --mail-user=insert_your_email_here
# Wall clock limit:
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE 
#SBATCH --exclusive

#Number of nodes and MPI tasks per node:
#SBATCH --ntasks=

##fixed frequency, no dynamic adjustment
###SBATCH --ear=off
#optional: keep job within one island
#SBATCH --switches=1



#Important

SLURM_CPU_BIND=verbose

SGPP_DIR=/home/pollinta/epyc/DisCoTec/
LIB_BOOST_DIR=
LIB_GLPK=$SGPP_DIR/glpk/lib/

export LD_LIBRARY_PATH=$SGPP_DIR/lib/sgpp:$LIB_GLPK:$LIB_BOOST_DIR:$LD_LIBRARY_PATH
. /home/pollinta/epyc/spack/share/spack/setup-env.sh
spack load boost@1.74.0
# for boost

# Change to the direcotry that the job was submitted from
# cd $PBS_O_WORKDIR

paramfile="ctparam"
# allows to read the parameter file from the arguments.
# Useful for testing the third level combination on a single system
if [ $# -ge 1 ] ; then
   paramfile=$1
fi

ngroup=$(grep ngroup $paramfile | awk -F"=" '{print $2}')
nprocs=$(grep nprocs $paramfile | awk -F"=" '{print $2}')

mpiprocs=$((ngroup*nprocs+1))

# thread pinning, cf. 
# https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux/top/environment-variable-reference/process-pinning/environment-variables-for-process-pinning.html
export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST=48-95
#export I_MPI_PIN_PROCESSOR_LIST=0-47
export I_MPI_PIN_PROCESSOR_LIST="allcores"
#"0-47:map=bunch"
export I_MPI_PIN_ORDER=compact

# General
#mpiexec -n "$mpiprocs" ./xthi
mpiexec -n "$mpiprocs" ./combi_example $paramfile
# Use for debugging
#mpirun -n "$mpiprocs" xterm -hold -e gdb -ex run --args ./combi_example $paramfile

