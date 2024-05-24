#!/bin/bash
#SBATCH --mail-user=vjadhao@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=2:00:00
#SBATCH -A r00312
#SBATCH --partition=general
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=EEE2c100_12cores
#SBATCH --output=out.log
#SBATCH --error=err.log

module load lammps/29Oct20

cd      $SLURM_SUBMIT_DIR
time srun -n 12 lmp_mpi < in.lammps.template
