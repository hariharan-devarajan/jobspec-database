#!/bin/bash
#SBATCH --mail-user=vjadhao@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=E2
#SBATCH --output=out.log
#SBATCH --error=err.log

module swap PrgEnv-intel PrgEnv-gnu
module load lammps/gnu

cd      $SLURM_SUBMIT_DIR
time srun -n 48 lmp_mpi < in.lammps.template
