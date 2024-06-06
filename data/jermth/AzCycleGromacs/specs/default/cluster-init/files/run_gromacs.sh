#!/bin/bash -l
#PBS -l mem=64GB
#PBS -l nodes=2:ppn=16

. /opt/intel/compilers_and_libraries/linux/mpi/intel64/bin/mpivars.sh
source /opt/intel/compilers_and_libraries_2017.2.174/linux/bin/compilervars.sh intel64
export I_MPI_FABRICS=dapl
export I_MPI_DAPL_PROVIDER=ofa-v2-ib0
export I_MPI_DEBUG=5

export GROMACS_HOME=/mnt/resource/gromacs-5.1.4
export PATH=$GROMACS_HOME/bin:$PATH

cd $PBS_O_WORKDIR

echo "#### ENV VAR FOR DEBUGGING #######"
env
echo "##################################"

echo "PBS_O_WORKDIR: $PBS_O_WORKDIR"
echo "PBS_NODEFILE: $PBS_NODEFILE"
echo "NODES LIST:"
cat $PBS_NODEFILE
echo ""
echo ""
echo "RUN"

mpirun -n 32 -ppn 16 -hostfile $PBS_NODEFILE  gmx_mpi grompp -f pme.mdp
