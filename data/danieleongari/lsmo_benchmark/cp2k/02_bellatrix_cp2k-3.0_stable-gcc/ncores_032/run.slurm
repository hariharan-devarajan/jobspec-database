#!/bin/bash -l
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --time=00:30:00

source /ssoft/spack/bin/slmodules.sh -r stable 

module purge
module load gcc/5.3.0 mvapich2/2.2b openblas/0.2.18 cp2k/3.0-mpi
export OMP_NUM_THREADS=1

date_start=$(date +%s)
srun cp2k.popt -i *.inp -o output.out                 #running command
date_end=$(date +%s)
time_run=$((date_end-date_start))
echo "032_cpus $time_run seconds"

rm -f GTH_BASIS_SETS POTENTIAL *xyz *restart          #remove useless big files (please!) 
