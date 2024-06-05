#!/bin/bash
#
#SBATCH --job-name=BCLUB_FPREP
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH --mem-per-cpu 80000
## %A == SLURM_ARRAY_JOB_ID
## %a == SLURM_ARRAY_TASK_ID
#SBATCH -o /projects/niblab/bids_projects/Experiments/BreakfastClub/error_files/fprep_%A_%a_ses-1_out.txt
#SBATCH -e /projects/niblab/bids_projects/Experiments/BreakfastClub/error_files/fprep_%a_ses-1_err.txt


if [ ${SLURM_ARRAY_TASK_ID} -lt 10 ]; then
    sub="sub-00${SLURM_ARRAY_TASK_ID}"
else
    sub="sub-0${SLURM_ARRAY_TASK_ID}"
fi

singularity exec -B /:/home_dir /projects/niblab/bids_projects/Singularity_Containers/fmriprep.simg fmriprep /home_dir/projects/niblab/bids_projects/Experiments/BreakfastClub/BID$
    participant  \
    --participant-label $sub \
    --fs-license-file /home_dir/projects/niblab/bids_projects/freesurfer/license.txt \
    --longitudinal \
    --fs-no-reconall \
    --omp-nthreads 16 --n_cpus 16  \
    --bold2t1w-dof 12 \
    --output-space template --template MNI152NLin2009cAsym \
    -w /home_dir/projects/niblab/bids_projects/Experiments/BreakfastClub/fmriprep/${sub}/ses-1 \
    --debug  \

