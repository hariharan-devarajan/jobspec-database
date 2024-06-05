#!/bin/bash -l
#
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --job-name=NewTitle
#SBATCH --partition=thsu
#SBATCH --time=7-00:00:00
#SBATCH --output=mylog.out
#SBATCH --error=myfail.out
#SBATCH --mail-user='rschanta@udel.edu'
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=ALL
#
#UD_QUIET_JOB_SETUP=YES
#UD_USE_SRUN_LAUNCHER=YES
#UD_DISABLE_CPU_AFFINITY=YES
#UD_MPI_RANK_DISTRIB_BY=CORE
#UD_DISABLE_IB_INTERFACES=YES
#UD_SHOW_MPI_DEBUGGING=YES
#
. /opt/shared/slurm/templates/libexec/openmpi.sh
. /work/thsu/rschanta/RTS/functions/utility/bash-utils.sh
#
vpkg_require openmpi
vpkg_require matlab
	run_MATLAB_script "make_FW2.m"
