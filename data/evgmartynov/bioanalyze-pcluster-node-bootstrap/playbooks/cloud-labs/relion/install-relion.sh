#!/usr/bin/env bash

#SBATCH -N 4
#SBATCH -n 4
#SBATCH --partition dev
#SBATCH --constraint m5a4xlarge
#SBATCH --time 1-00:00:00
#SBATCH --exclusive

# First try to install with the binary cache
srun -v -v -N 4 -n 4 spack install \
  -v -y \
  --deprecated \
  --no-check-signature \
  --no-checksum \
  --use-cache \
  relion@3.1.3 \
  relion@4.0-beta

spack install \
  --no-check-signature \
  --no-checksum \
  --use-cache \
  relion@3.1.3 ~mklfft ~cuda \
  relion@4.0-beta ~mklfft ~cuda \
  relion@3.1.3 ~mklfft +cuda \
  relion@4.0-beta ~mklfft +cuda

