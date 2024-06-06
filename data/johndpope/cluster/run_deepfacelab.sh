#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH

#SBATCH --partition gtx1080                         ### specify partition name where to run a job. NVidia 2080: short - 7 days; 1080: gtx1080 - 7 days; debug – for testing - 2 hours and 1 job at a time
#SBATCH --time 0-07:00:00                           ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name run_deepfacelab                 ### name of the job. replace my_job with your desired job name
#SBATCH --output run_deepfacelab-id-%J.out                 ### output log for running job - %J is the job number variable
#SBATCH --mail-user=chenmis@post.bgu.ac.il          ### users email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL                  ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gres=gpu:1                                ### number of GPUs (can't exceed 8 gpus for now) ask for more than 1 only if you can parallelize your code for multi GPU


### Print some data to output file ###

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST


### Start you code below ####

module load anaconda                          ### load anaconda module
source activate py37   

bash -i ./2_extract_image_from_data_src.sh
