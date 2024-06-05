#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=2:00:00
#SBATCH --mem=5GB
#SBATCH --job-name=equilibrate_pressure
#SBATCH --mail-type=END
#SBATCH --mail-user=tje3676@nyu.edu

module purge
module load gromacs/openmpi/intel/2020.4

# move into directory
cd /home/tje3676/chem-class-2023/comp-lab-class/Week3/Data

# Run NPT
gmx_mpi mdrun -deffnm npt
