#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=3GB
#SBATCH --mail-type=END
#SBATCH --mail-user=tje3676@nyu.edu
#SBATCH --job-name=structure_B

module purge
source /scratch/work/courses/CHEM-GA-2671-2023fa/software/gromacs-2019.6-plumedSept2020/bin/GMXRC.bash.modules

# Run gromacs
gmx_mpi mdrun -s topolB.tpr -nsteps 10000000
