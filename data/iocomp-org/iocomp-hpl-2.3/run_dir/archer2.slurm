#!/bin/bash --login
#SBATCH --job-name=hpl
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --account=e609
#SBATCH --partition=standard 
#SBATCH --qos=lowpriority

# ARCHER2 module loads 
module swap PrgEnv-cray/8.0.0 PrgEnv-gnu
module use /work/y07/shared/archer2-lmod/dev
module load cray-hdf5-parallel
module load cmake 
module swap craype-network-ofi craype-network-ucx 
module swap cray-mpich cray-mpich-ucx 

# Setup environment
export PPN=${SLURM_NTASKS_PER_NODE}
export OMP_NUM_THREADS=1
export IOCOMP_DIR=/work/e609/e609/shr203/iocomp
export LD_LIBRARY_PATH=${IOCOMP_DIR}/lib:${LD_LIBRARY_PATH} 
export ADIOS2_DIR=/work/e609/e609/shr203/opt/gnu/8.0.0/ADIOS2
export LD_LIBRARY_PATH=${ADIOS2_DIR}/lib64:${LD_LIBRARY_PATH} 
export HPL=/mnt/lustre/a2fs-work3/work/e609/e609/shr203/hpl-2.3/bin/cray/xhpl
export CONFIG=${SLURM_SUBMIT_DIR}/config.xml 

#match HPL.dat file for the correct number of processes with the correct prefilled distribution between P and Q
NUM_COMP_PROCS=$((${SLURM_NNODES} * ${SLURM_NTASKS_PER_NODE}/2 )) # as number of compute processes are half of num nodes
HPL_DAT=${SLURM_SUBMIT_DIR}/dat_files/${NUM_COMP_PROCS}.dat 

# Make new directory 
IOLAYERS=("MPIIO" "HDF5" "ADIOS2_HDF5" "ADIOS2_BP4" "ADIOS2_BP5") # assign IO layer array 
i=${SLURM_ARRAY_TASK_ID} 
HALF_CORES=$((${SLURM_NTASKS_PER_NODE}/2)) 
FULL_CORES=$((${SLURM_NTASKS_PER_NODE})) 
HALF_NODES=$((${SLURM_NNODES}/2))
MAP=0

echo "Job started " $(date +"%T") # start time

for m in $(seq ${IO_start} ${IO_end})
do 
  export PARENT_DIR=${SLURM_SUBMIT_DIR}/${DIR}/${SLURM_NNODES}_${SLURM_NTASKS_PER_NODE}/${IOLAYERS[${m}]}

  # Case 1 
	source ${SLURM_SUBMIT_DIR}/slurm_files/sequential.sh
	wait 

#  # Case 2
#	source ${SLURM_SUBMIT_DIR}/slurm_files/oversubscribe.sh 
#	wait 

 # Case 3
 source ${SLURM_SUBMIT_DIR}/slurm_files/hyperthread.sh 
 wait 

  # Case 4
	source ${SLURM_SUBMIT_DIR}/slurm_files/consecutive.sh
	wait 
done 

echo $(module list) 

# runtime given by: 
echo "Job ended " $(date +"%T") # end time 
