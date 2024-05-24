#!/bin/bash
#SBATCH -J lammps
#SBATCH --partition=normal12
#SBATCH --ntasks=12

INPUT_FILE='fix_fcc_Cu.in'

export OMP_NUM_THREADS=1

echo "--- STARTING JOB SCRIPT -----------------------------------------"
echo "-> Loading Modules"
module load intel_parallel_studio_xe/2020.1

echo "-> Working directory"
pwd
echo "--- RUNNING PI     ----------------------------------------------"
mpirun -n 12 $HOME/bin/lmp < $INPUT_FILE

echo "--- FINISHING JOB SCRIPT ----------------------------------------"
