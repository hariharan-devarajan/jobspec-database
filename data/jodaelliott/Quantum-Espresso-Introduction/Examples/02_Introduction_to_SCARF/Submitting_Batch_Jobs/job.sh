#!/bin/bash
#SBATCH -p scarf
#SBATCH -C amd
#SBATCH --nodes 1 --ntasks-per-node=32
#SBATCH -t 10:00
#SBATCH -o job_%J.log
#SBATCH -e job_%J.err

export OMP_NUM_THREADS=1
module load contrib/dls-spectroscopy/quantum-espresso/7.3.1-GCC-12.2.0

mpirun -np ${SLURM_NTASKS} pw.x -inp c60_scf.pwi > c60_scf.pwo
