#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=3GB
#SBATCH --mail-type=END
#SBATCH --mail-user=tje3676@nyu.edu
#SBATCH --job-name=prepare files

module purge
module load gromacs/openmpi/intel/2018.3

# Run gromacs
gmx_mpi grompp -f adp_T300.mdp -c adp.gro -p adp.top -o T300/adp.tpr
