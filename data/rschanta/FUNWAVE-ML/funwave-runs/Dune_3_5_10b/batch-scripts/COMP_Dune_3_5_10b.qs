#!/bin/bash -l
#
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --job-name=COMP_Dune_3_5_10b
#SBATCH --partition=standard
#SBATCH --time=7-00:00:00
#SBATCH --output=./Dune_3_5_10b/slurm_logs/COMP_out.out
#SBATCH --error=./Dune_3_5_10b/slurm_logs/COMP_err.out
#SBATCH --mail-user=rschanta@udel.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=ALL
#SBATCH --dependency=afterany:27624300
#
#UD_QUIET_JOB_SETUP=YES
#UD_USE_SRUN_LAUNCHER=YES
#UD_DISABLE_CPU_AFFINITY=YES
#UD_MPI_RANK_DISTRIB_BY=CORE
#UD_DISABLE_IB_INTERFACES=YES
#UD_SHOW_MPI_DEBUGGING=YES
#
#
## Load in utilities and VALET
. "/work/thsu/rschanta/RTS/functions/bash-utility/slurm-bash.sh"
. "/work/thsu/rschanta/RTS/functions/bash-utility/matlab-bash.sh"
. "/work/thsu/rschanta/RTS/functions/bash-utility/misc-bash.sh"
	vpkg_require matlab

## Compress outputs from all runs to a single structure
	run_compress_out /lustre/scratch/rschanta/ Dune_3_5_10b /work/thsu/rschanta/RTS/

## Keep for now
	#rm -rf "/lustre/scratch/rschanta/Dune_3_5_10b/outputs-proc/"
	rm -rf "/lustre/scratch/rschanta/Dune_3_5_10b/outputs-raw/"
