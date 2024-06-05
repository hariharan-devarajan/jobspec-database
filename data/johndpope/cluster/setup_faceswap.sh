#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like ##SBATCH

#SBATCH --partition gtx1080                         ### specify partition name where to run a job. NVidia 2080: short - 7 days; 1080: gtx1080 - 7 days; debug – for testing - 2 hours and 1 job at a time
#SBATCH --time 0-07:00:00                           ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name run_faceswap                 ### name of the job. replace my_job with your desired job name
#SBATCH --output run_faceswap-id-%J.out                 ### output log for running job - %J is the job number variable
#SBATCH --mail-user=chenmis@post.bgu.ac.il          ### users email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL                  ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gres=gpu:1                                ### number of GPUs (can't exceed 8 gpus for now) ask for more than 1 only if you can parallelize your code for multi GPU


### Print some data to output file ###

echo "SLURM_JOBID"=$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST


### Start you code below ####

module load anaconda                          ### load anaconda module
source activate py37                          ### activating environment, environment must be configured before running the job
### pip install --user tensorflow==2.2 ###
### pip install --user opencv-python ### 
### pip install --user pynvml ###
### pip install --user imageio_ffmpeg ###
### pip install --user fastcluster ### 
### pip install --user pathlib ###

### python ../faceswap/faceswap.py extract -i take000.mp4 -o src000 ###
### python ../faceswap/faceswap.py extract -i take002.mp4 -o src002 ###
### python ../faceswap/faceswap.py train -A src000 -B src002 -m train ###
### python ../faceswap/faceswap.py convert -i src000 -o converted -m train ###
### python ../faceswap/faceswap.py extract -h ###
### python ../faceswap/faceswap.py train -h ###
### python ../faceswap/faceswap.py convert -h ###

mkdir Project
mkdir Project/faceA
mkdir faceSwapProject/faceA
mkdir faceSwapProject/modelAB
mkdir faceSwapProject/TimelapseAB

python ../faceswap/faceswap.py extract -i Project/src/faceA.mp4 -o Project/faceA -D s3fd -A fan 
python ../faceswap/faceswap.py extract -i Project/src/faceB.mp4 -o Project/faceB -D s3fd -A fan 

python ../faceswap/faceswap.py train -A Project/faceA -ala Project/src/faceA_alignments.fsa -alb Project/src/faceB_alignments.fsa -B Project/faceB -m Project/ModelAB -t original -tia Project/faceA -tib Project/faceB -to Project/TimelapseAB

python ../faceswap/faceswap.py convert -i Project/src/faceA.mp4 -o Project -m Project/ModelAB -c match-hist -M extended -w ffmpeg 

