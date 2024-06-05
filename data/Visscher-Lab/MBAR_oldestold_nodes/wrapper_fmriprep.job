#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000
#SBATCH --partition=express
#SBATCH --time=2:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=snolin@uab.edu
#SBATCH --job-name=Fmriprep_submitjob
#SBATCH --output=Fmriprep_submitjob.txt

#this script runs fmriprep
#pre run freesurfer need to be in derivatives folder if you want them run with fmriprep and must have "sub-" prefix
#if freesurfer folder has scripts/isrunning file fmriprep will error out

#folder for job files to go
jobs=/data/project/vislab/a/MBAR/Resting_preproc/fmriprep_jobs
logs=/data/project/vislab/a/MBAR/Resting_preproc/fmriprep_logs
mkdir $jobs
mkdir $logs
#list of subjects you want to pull
D=/data/project/vislab/a/MBAR/Anat_preproc/AllSites_BIDS/

for patient in `ls -1 $D`
do
#patient=`echo $patient | cut -c5-14`;
echo "#!/bin/bash
#SBATCH --partition=long
#SBATCH --time=100:00:00
#SBATCH --job-name=fmriprep_$patient
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=25000
#SBATCH --output=$logs/fmriprep_$patient.out
module load Singularity/2.5.2-GCC-5.4.0-2.26 
export SINGULARITY_BINDPATH=/data/project/vislab
cd /data/project/vislab/MISC/Scripts/fmriprep/
singularity run fmriprep-1.2.5.simg --output-space T1w template --verbose --fs-license license.txt --skip_bids_validation -t rest --n_cpus 5 --omp-nthreads 5 -w /data/project/vislab/a/MBAR/Resting_preproc/workdir /data/project/vislab/a/MBAR/Anat_preproc/AllSites_BIDS/ /data/project/vislab/a/MBAR/Anat_preproc/derivatives/ participant --participant_label $patient" > $jobs/fmriprep_$patient.job
#output /data/project/vislab/a/MBAR/Anat_preproc/derivatives/ participant --participant_label $patient
#Bids data /data/project/vislab/a/MBAR/Anat_preproc/AllSites_BIDS/
sbatch $jobs/fmriprep_$patient.job
done
