#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --time=24:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
# ------------------------------------------

module load singularity/3.5.3
module load freesurfer/7.2.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh
subject_name=`sed "${SLURM_ARRAY_TASK_ID}q;d" subj.txt`

STUDY=/work/cnelab/TECHS/MRI/BIDS
BIDS_DIR="${STUDY}"
DERIVS_DIR=/derivatives/fmriprep-23.0.2
LOCAL_FREESURFER_DIR="${BIDS_DIR}/derivatives/freesurfer_7.2.0/post"
CONTAINER_PATH=/shared/container_repository/fmriprep/23.0.2
CONTAINER_NAME=fmriprep_23.0.2.sif

# Prepare some writeable bind-mount points.
TEMPLATEFLOW_HOST_HOME=/scratch/$USER/fmriprep_work/.cache/templateflow
FMRIPREP_HOST_CACHE=/scratch/$USER/fmriprep_work/.cache/fmriprep
mkdir -p ${TEMPLATEFLOW_HOST_HOME}
mkdir -p ${FMRIPREP_HOST_CACHE}

# Prepare derivatives folder
mkdir -p ${BIDS_DIR}/${DERIVS_DIR}
mkdir -p ${LOCAL_FREESURFER_DIR}

echo "Freesurfer has been loaded and set up, the license can be found here $FREESURFER_HOME/license.txt"

# Make sure FS_LICENSE is defined in the container.
export SINGULARITYENV_FS_LICENSE=$FREESURFER_HOME/license.txt

# Designate a templateflow bind-mount point
export SINGULARITYENV_TEMPLATEFLOW_HOME="/templateflow"
SINGULARITY_CMD="singularity run --cleanenv -B /shared:/shared -B $BIDS_DIR:/data -B ${TEMPLATEFLOW_HOST_HOME}:${SINGULARITYENV_TEMPLATEFLOW_HOME} -B /scratch/$USER/my_fmriprep_post:/work -B ${LOCAL_FREESURFER_DIR}:/fsdir $CONTAINER_PATH/$CONTAINER_NAME"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
#subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# Remove IsRunning files from FreeSurfer
#find ${LOCAL_FREESURFER_DIR}/sub-Pilot_1/ -name "*IsRunning*" -type f -delete

# Compose the command line
cmd="${SINGULARITY_CMD} /data /data/${DERIVS_DIR} participant --participant-label $subject_name -w /work/ -vv --omp-nthreads 8 --nthreads 12 --mem_mb 30000 --output-spaces MNI152NLin2009cAsym:res-2 anat fsnative fsaverage5 --use-aroma --fs-subjects-dir /fsdir --skip_bids_validation"
# Setup done, run the command
echo Running task ${SLURM_ARRAY_TASK_ID}
echo Commandline: $cmd
eval $cmd
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> ${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv
echo $subject_name
echo Finished tasks ${SLURM_ARRAY_TASK_ID} with exit code $exitcode
