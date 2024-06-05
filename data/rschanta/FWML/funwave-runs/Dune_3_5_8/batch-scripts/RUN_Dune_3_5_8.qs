#!/bin/bash -l
#
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --job-name=RUN_Dune_3_5_8
#SBATCH --partition=standard
#SBATCH --time=7-00:00:00
#SBATCH --output=./Dune_3_5_8/slurm_logs/RUN_out_%a.out
#SBATCH --error=./Dune_3_5_8/slurm_logs/RUN_err_%a.out
#SBATCH --mail-user=rschanta@udel.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=ALL
#SBATCH --dependency=afterany:27620480
#SBATCH --array=1-2
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
	. /opt/shared/slurm/templates/libexec/openmpi.sh
	vpkg_require openmpi
	vpkg_require matlab

## FUNWAVE Executable Path
	fun_ex="/work/thsu/rschanta/RTS/funwave/v3.6/exec/FW-COUP"
	echo "$fun_ex"

## Get input file name
	input_file=$(get_input_dir "/lustre/scratch/rschanta/" "Dune_3_5_8" "$SLURM_ARRAY_TASK_ID")

## Run FUNWAVE
	${UD_MPIRUN} "$fun_ex" "$input_file"

## Compress outputs from run to single structure
	run_compress_out_ska_i /lustre/scratch/rschanta/ Dune_3_5_8 "$SLURM_ARRAY_TASK_ID" "/work/thsu/rschanta/RTS/"

