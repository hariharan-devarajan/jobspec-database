#!/bin/bash
#SBATCH	--nodes=1
#SBATCH -J deepcam-cgpu
#SBATCH -C gpu
#SBATCH -A nstaff
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=10
#SBATCH --dependency=singleton
#SBATCH --time 2:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Run parameters
export BATCHSIZE=2
export DO_PROFILING='false'  # true or false
export DO_NCCL_DEBUG='false' # true or false

# Setup software environment
module load cgpu
module load pytorch/v1.6.0-gpu
export NODES=${SLURM_NNODES}

if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
    srun -N $SLURM_NNODES -n $((SLURM_NNODES*8)) -c 10 \
         --cpu-bind=cores \
         ./utils/run_with_profiling.sh
else
    srun -N $SLURM_NNODES -n $((SLURM_NNODES*8)) -c 10 \
         --cpu-bind=cores \
         ./utils/run.sh
fi
