#!/bin/bash
#
#SBATCH --array=0-{{{max_node}}}
#SBATCH --job-name={{{jobname}}}
#SBATCH --output={{{project_dir}}}/{{{tmp_dir}}}/slurm_%a.out
{{#flags}}
#SBATCH --{{{name}}}
{{/flags}}
{{#options}}
#SBATCH --{{{name}}}={{{value}}}
{{/options}}
#SBATCH --chdir={{{project_dir}}}

# Load the associated singularity R module
module load simg_R/{{{r_version}}}

# Set the SLURM_ARRAY_TASK_ID variable to be exported to singularity
export SINGULARITYENV_SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}

# Run the rscript
Rscript {{{tmp_dir}}}/slurm_run.R
