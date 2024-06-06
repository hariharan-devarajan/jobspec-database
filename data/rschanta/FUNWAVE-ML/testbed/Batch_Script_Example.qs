#!/bin/bash -l
#
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --job-name=ExampleBatchScript
#SBATCH --partition=thsu
#SBATCH --time=7-00:00:00
#SBATCH --output=output.out
#SBATCH --error=error.out
#SBATCH --mail-user=rschanta@udel.edu
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
#
## Load in VALET
	vpkg_require matlab
	vpkg_require openmpi
. /opt/shared/slurm/templates/libexec/openmpi.sh

## Run Generation Script
	echo "Hello World!"
