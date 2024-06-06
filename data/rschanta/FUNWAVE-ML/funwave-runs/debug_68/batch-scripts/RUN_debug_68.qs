#!/bin/bash -l
#
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --job-name=RUN_
#SBATCH --partition=standard
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/thsu/rschanta/RTS/funwave-runs/debug_68/slurm_logs/RUN_out_%a.out
#SBATCH --error=/work/thsu/rschanta/RTS/funwave-runs/debug_68/slurm_logs/RUN_err_%a.out
#SBATCH --mail-user=rschanta@udel.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --export=ALL
#SBATCH --dependency=afterany:27850975
#SBATCH --array=1-4
#
#UD_QUIET_JOB_SETUP=YES
#UD_USE_SRUN_LAUNCHER=YES
#UD_DISABLE_CPU_AFFINITY=YES
#UD_MPI_RANK_DISTRIB_BY=CORE
#UD_DISABLE_IB_INTERFACES=YES
#UD_SHOW_MPI_DEBUGGING=YES
#
#
		## Load in bash functions and VALET packages
			export WORK_DIR=/work/thsu/rschanta/RTS/
			. "/work/thsu/rschanta/RTS/functions/bash-utility/get_bash.sh"
			export_vars "/lustre/scratch/rschanta/" "/work/thsu/rschanta/RTS/" "debug_68" "rschanta@udel.edu"
			vpkg_require openmpi
			vpkg_require matlab
		## Get input file name
			input_file=$(get_input_dir "$SLURM_ARRAY_TASK_ID")
		## Run FUNWAVE
			${UD_MPIRUN} "/work/thsu/rschanta/RTS/funwave/v3.6/exec/FW-REG" "$input_file"
		## Compress outputs from run to single structure, calculate skew and asymmetry too
			run_comp_i "$SLURM_ARRAY_TASK_ID" "{'new_ska'}"
		## Delete raw files from run
			rm_raw_out_i "$SLURM_ARRAY_TASK_ID"
