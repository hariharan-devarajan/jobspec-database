#!/bin/bash

#SBATCH --job-name=freesurfer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50G
#SBATCH --gres=gpu:tesla:1
#SBATCH --mail-user=l.liu9@uq.edu.au
#SBATCH --mail-type=ALL

module load cuda/9.0.176.1
module load mvapich2-gnu4/2.3
module load fftw-3.3.7-gcc-4.8.5-5igtfs5
module load freesurfer/6.0.0

# source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/afm01/Q1/Q1391/MRI_segmentation/free_surfer_data
echo $SUBJECTS_DIR
# export FS_LICENSE=/afm01/Q1/Q1391/MRI_segmentation/pytorch-3dunet/license.txt
# echo $FS_LICENSE

start_time="$(date -u +%s)"
# recon-all -parallel -openmp 8 -i /afm01/Q1/Q1391/MRI_segmentation/ATLAS_R1.1/Site1/031768/t01/031768_t1w_deface_stx.nii/031768_t1w_deface_stx.nii -all -subjid 031768_t1w_deface_stx
recon-all -parallel -openmp 32 -i "$SUBJECTS_DIR"/031769_t1w_deface_stx.nii.gz -all -subjid 031769_t1w_deface_stx
end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total of $elapsed seconds elapsed for process"
