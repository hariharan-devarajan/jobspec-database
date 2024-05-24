#!/bin/bash

#SBATCH --job-name=myFirstMPIJob
#SBATCH --output=myOutput-n1.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=multithread
#SBATCH --mem-per-cpu=100M
#SBATCH --time=00:01:00
#SBATCH --partition=cpu-core-0
#SBATCH --account=cpu-s5-grad_778-1
#SBATCH --mail-user=YOUREMAILHERE
#SBATCH --mail-type=ALL

export OMP_NUM_THREADS=2

module load openmpi/gcc/4.0.4 
module load singularity 

mpirun singularity exec lammps_latest.sif lmp_mpi < input.lammps


