#!/bin/bash

#SBATCH --job-name=run-gromacs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=72:00:00
#SBATCH --mem=20GB

cd /scratch/work/courses/CHEM-GA-2671-2022fa/yw5806/comp-lab-class/Inputs/ParallelTemp

module purge

module load gromacs/openmpi/intel/2018.3


mpirun -np 4 gmx_mpi mdrun -s adp -multidir T300/ T350 T400/ T450 -deffnm adp_exchange4temps -replex 50