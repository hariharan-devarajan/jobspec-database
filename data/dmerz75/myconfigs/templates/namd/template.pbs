#!/bin/bash
#PBS -l nodes=1:ppn=xxprocsxx:xxnodexx
#PBS -l walltime=250:00:00
#PBS -N xxsubnamexx

# Fix environment variables.
USER=xxuserxx # self.user
MY_DIR=xxmy_dirxx # self.cwd
JOB_DIR_NAME=xxjob_dir_namexx # self.job_dir_name
JOB_PATH_LOCAL=xxjob_path_localxx # self.job_path_local
JOB_PATH_RUN=xxjob_path_runxx # self.job_path_run # /scratch/user/$JOB_SUB
JOB_COMPLETED_DIR=xxcompleted_dirxx # self.completed_dir # /data4/user/completed/$JOB_SUB

# Prepare directories.
rm -rf $JOB_PATH_RUN
mkdir -p $JOB_PATH_RUN
cp $JOB_PATH_LOCAL/* $JOB_PATH_RUN
cd $JOB_PATH_RUN

# Job Execution.
NAMD_DIR=NAMD_2.9_Linux-x86_64-multicore
${HOME}/opt/${NAMD_DIR}/namd2 +pxxprocsxx xxnamdconfigxx > run.log &
wait

# Clean up!
rsync -auvz $JOB_PATH_RUN/* $JOB_COMPLETED_DIR
#rm -rf $JOB_DIR
