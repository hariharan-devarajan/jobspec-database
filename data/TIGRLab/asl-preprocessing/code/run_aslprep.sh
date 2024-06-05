#!/bin/bash -l

#SBATCH --partition=low-moby
#SBATCH --array=1-154
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=6:00:00
#SBATCH --job-name aslprep
#SBATCH --output=aslprep_%j.out
#SBATCH --error=aslprep_%j.err

STUDY="TAY"

sublist="/scratch/mjoseph/asl-preprocessing/code/subject_list.txt"

index() {
   head -n $SLURM_ARRAY_TASK_ID $sublist \
   | tail -n 1
}

BIDS_DIR=/archive/data/${STUDY}/data/bids
OUT_DIR=/archive/data/${STUDY}/pipelines/in_progress
DANAT_DIR=/scratch/jwong/fmriprep_dl/fmriprep
CODE_DIR=/scratch/mjoseph/asl-preprocessing/code
TMP_DIR=/scratch/mjoseph/tmp
WORK_DIR=${TMP_DIR}/${STUDY}/aslprep
FS_LICENSE=${TMP_DIR}/freesurfer_license/license.txt

SING_CONTAINER=/archive/code/containers/ASLPREP/pennlinc_aslprep_0.2.8-2022-01-19-97e5866ebcbc.simg

mkdir -p $BIDS_DIR $OUT_DIR $TMP_DIR $WORK_DIR

singularity run \
  -H ${TMP_DIR} \
  -B ${BIDS_DIR}:/bids \
  -B ${OUT_DIR}:/out \
  -B ${DANAT_DIR}:/danat \
  -B ${CODE_DIR}:/code \
  -B ${WORK_DIR}:/work \
  -B ${FS_LICENSE}:/li \
  -B /projects/mjoseph/pipelines/aslprep/aslprep:/usr/local/miniconda/lib/python3.7/site-packages/aslprep \
  ${SING_CONTAINER} \
  /bids /out participant \
  --skip-bids-validation \
  --participant_label `index` \
  --bids-filter-file /code/filter_aslprep.json \
  --n_cpus 4 \
  --anat-derivatives /danat \
  --smooth_kernel 0 \
  --m0_scale 100 \
  --output-spaces MNI152NLin2009cAsym asl
  -w /work \
  --notrack
