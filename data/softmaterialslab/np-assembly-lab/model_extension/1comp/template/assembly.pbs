#!/bin/bash
#SBATCH --mail-user=cfaccini@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=3:00:00
#SBATCH -A r00312
#SBATCH --partition=general
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=USERVLP_USERSALTCONC
#SBATCH --output=out.log
#SBATCH --error=err.log

module load lammps/29Oct20

cd      $SLURM_SUBMIT_DIR
time srun -n 48 lmp_mpi < in.1comp.template
