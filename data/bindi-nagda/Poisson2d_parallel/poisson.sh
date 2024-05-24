#!/bin/bash
#SBATCH --job-name 2D
#SBATCH --output=Poisson2d.%J.out
#SBATCH --error=Poisson2d_16.%J.err
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=5000MB
#SBATCH --partition=long
mpirun -np ntasks ./Poisson2D -da_grid_x 256 -da_grid_y 256 -pc_type gamg 

