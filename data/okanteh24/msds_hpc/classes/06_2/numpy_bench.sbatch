#!/bin/bash
#SBATCH -J numpy
#SBATCH -o numpy_%A_%a.out
#SBATCH -p development
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --array=0,1
#SBATCH -x k001

module purge
module load singularity

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

img=("spack_python_20.04_openblas-2022-06-02-cded0171328d.simg"\
     "spack_python_20.04_mkl-2022-06-02-ac0ca4acdb94.simg")

echo "Using ${img[${SLURM_ARRAY_TASK_ID}]} on ${HOSTNAME}."

singularity exec ${img[${SLURM_ARRAY_TASK_ID}]} /opt/view/bin/python3 gemm.py

