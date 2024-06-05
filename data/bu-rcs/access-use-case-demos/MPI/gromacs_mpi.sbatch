#!/bin/bash
#SBATCH -J gmx_mpi_test              # job name
#SBATCH -e gmx_mpi_test.%j.out       # error file name
#SBATCH -o gmx_mpi_test.%j.out       # output file name
#SBATCH -N 2                  # request 2 nodes
#SBATCH -n 96                 # request 2x48=96 MPI tasks
#SBATCH -p skx                # designate queue
#SBATCH -t 24:00:00           # designate max run time
#SBATCH -A myproject          # charge job to myproject

module load intel/24.0
module load impi/21.11
module load gromacs/2023.3

ibrun gmx_mpi pdb2gmx -f 1AKI_clean.pdb -o 1AKI_processed.gro -water spce -ff oplsaa