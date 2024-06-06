#!/bin/bash
#SBATCH -J TUNE
#SBATCH -A uoa00035 # Project Account
#SBATCH --time=5:00:00     # Walltime
##SBATCH --ntasks=48          # number of tasks
##SBATCH --tasks-per-node=24
#SBATCH --mem-per-cpu=2048   # memory/cpu (in MB)
##SBATCH -C ib
#SBATCH -e stderr.txt
#SBATCH -o stdout.txt

#module use /projects/uoa00035/privatemodules
ml Python/3.4.3-intel-2015a
ml VTune/2015_update2
ml Delft3D/5128-intel-2015a
ml impi/5.0.3.048-iccifort-2015.2.164-GCC-4.9.2

ml itac/9.0.3.051
source itacvars.sh impi5

unset I_MPI_PMI_LIBRARY #required
export I_MPI_FABRICS=shm:dapl

export LANG=C
export LC_ALL=C
export LC_CTYPE=C
mpitune -a \"mpiexec.hydra d_hydro.exe config_d_hydro.xml\" -of tune.conf 


