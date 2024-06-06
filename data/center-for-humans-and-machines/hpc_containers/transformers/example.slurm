#!/bin/bash -l
#
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J transformers
#
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.carreto.fidalgo@gmail.com
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=00:15:00

source /etc/profile.d/modules.sh
module purge
module load apptainer

# Avoid hyper-threading (in this case cpus-per-task // nr_of_gpus):
export OMP_NUM_THREADS=18

# For pinning threads correctly:
export OMP_PLACES=cores

# Useful for debugging:
# export NCCL_DEBUG=INFO


srun apptainer exec \
	--nv -B .:"$HOME" \
	transformers.sif torchrun \
		--nnodes="$SLURM_NNODES" \
		--nproc-per-node=gpu \
		--rdzv-id="$SLURM_JOBID" \
		--rdzv-endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
		--rdzv-backend="c10d" \
		example.py --lr=4e-5 --bs=2

