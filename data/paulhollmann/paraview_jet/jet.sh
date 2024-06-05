#!/bin/sh
#SBATCH -o ./pvscript.%j.%N.out
#SBATCH -D .
#SBATCH -J pvscript
#SBATCH --get-user-env
#SBATCH --partition=test
#SBATCH --nodes=16                            # what you need
#SBATCH --ntasks-per-node=48                 # what you need
#SBATCH --mail-type=none
#SBATCH --export=NONE
#SBATCH --time=00:10:00                      # what you need
#SBATCH --account=pn68pi              # necessary on SuperMUC-NG only!
 
module load slurm_setup                      # necessary workaround on SuperMUC-NG!
module load paraview-prebuild/5.8.0_mesa     # Look for available modules! But use MESA!
mpiexec pvbatch main.py       # try srun or use mpiexec's "-laucher" options if this does not work out-of-the-box