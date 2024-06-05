#!/bin/bash -x
#SBATCH --nodes=20
#SBATCH --ntasks=1360
#SBATCH --account=jiek63
#SBATCH --ntasks-per-node=68
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=20:00:00
#SBATCH --partition=booster

module --force purge
module load Architecture/KNL
module load intel-para

srun /p/project/cjiek63/jiek6304/JURECA_PFLOTRAN_280119/pflotran/src/pflotran/pflotran -pflotranin input_Permafrost.in