#!/bin/bash

#SBATCH --job-name=spack-install
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#####Use during training###############
#SBATCH --partition=reservation
#SBATCH --reservation=bootcamp_cpu_2023
#######################################

#####Use after training##############
##SBATCH --partition=express

## Load python module rwquired for spack
module load python/3.8.1

## Source spack environmnet:
source ~/spack/share/spack/setup-env.sh

## Install a package (uncomment this line for a first-time install):
#spack install tcl@8.6.12

## Source Spack again, and load the package:
source ~/spack/share/spack/setup-env.sh
spack load tcl

# Print the PATH
echo $PATH
