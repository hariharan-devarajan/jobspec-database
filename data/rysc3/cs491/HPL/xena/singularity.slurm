#!/bin/bash

#SBATCH -N 2
#SBATCH -G 2
#SBATCH -p singleGPU

module load  gcc/10.2.0-3kjq
module load singularity/3.7.0-bm53

export CONT=$(pwd)/nvidia-benchmarks-24.03.sif
export MOUNT=$(pwd)/HPL.dat

srun singularity run --nv -B "${MOUNT}" "${CONT}" ./hpl.sh --dat HPL.dat
