#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --account=cmse
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem-per-cpu=6G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name SimCLR      # you can give your job a name for easier identification (same as -J)
#SBATCH --time=3:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --cpus-per-task=5           # number of CPUs (or cores) per task (same as -c)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --gres=gpu:v100s:2
#SBATCH -o /mnt/home/renjie3/Documents/unlearnable/diffusion/diffusers/examples/text_to_image/logfile/%j.log
#SBATCH -e /mnt/home/renjie3/Documents/unlearnable/diffusion/diffusers/examples/text_to_image/logfile/%j.err

########## Command Lines for Job Running ##########

module purge
module load GCC/6.4.0-2.28 OpenMPI  ### load necessary modules.
conda activate diffuser

MY_ROOT_PATH="/mnt/home/renjie3/Documents/unlearnable/diffusion/diffusers/examples/text_to_image/"

cd ${MY_ROOT_PATH}
JOB_INFO="train stepsize"
MYCOMMEND="python train_text_to_image.py --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --train_data_dir=./haoyu/dataset --use_ema --resolution=512 --center_crop --random_flip --train_batch_size=6 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=2000 --learning_rate=5e-06 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=0 --output_dir=results/sd-haoyu-model --checkpointing_steps 100 --job_id ${SLURM_JOB_ID}_1"
MYCOMMEND2="No_commend2 --job_id ${SLURM_JOB_ID}_2"
MYCOMMEND3="No_commend3 --job_id ${SLURM_JOB_ID}_3"

#print the information of a job into one file
date >>${MY_ROOT_PATH}submit_history.log
echo $SLURM_JOB_ID >>${MY_ROOT_PATH}submit_history.log
echo $JOB_INFO >>${MY_ROOT_PATH}submit_history.log
echo $MYCOMMEND >>${MY_ROOT_PATH}submit_history.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${MY_ROOT_PATH}submit_history.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${MY_ROOT_PATH}submit_history.log
fi
echo "---------------------------------------------------------------" >>${MY_ROOT_PATH}submit_history.log

echo $JOB_INFO

echo $MYCOMMEND
$MYCOMMEND 1>${MY_ROOT_PATH}logfile/${SLURM_JOB_ID}_1.log 2>${MY_ROOT_PATH}logfile/${SLURM_JOB_ID}.err &

if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2
    $MYCOMMEND2 1>${MY_ROOT_PATH}logfile/${SLURM_JOB_ID}_2.log 2>${MY_ROOT_PATH}logfile/${SLURM_JOB_ID}_2.err &
fi

if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3
    $MYCOMMEND3 1>${MY_ROOT_PATH}logfile/${SLURM_JOB_ID}_3.log 2>${MY_ROOT_PATH}logfile/${SLURM_JOB_ID}_3.err &
fi
###python main.py --batch_size 512 --epochs 1000 --arch resnet18

wait

echo -e "JobID:$SLURM_JOB_ID \n JOB_INFO: ${JOB_INFO} \n Python_command: \n ${MYCOMMEND} \n ${MYCOMMEND2} \n ${MYCOMMEND3} \n " | mail -s "[Done] ${SLURM_JOB_ID}" thurenjie@outlook.com

date >>${MY_ROOT_PATH}finish_history.log
echo $SLURM_JOB_ID >>${MY_ROOT_PATH}finish_history.log
echo $JOB_INFO >>${MY_ROOT_PATH}finish_history.log
echo $MYCOMMEND >>${MY_ROOT_PATH}finish_history.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${MY_ROOT_PATH}finish_history.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${MY_ROOT_PATH}finish_history.log
fi
echo -e "---------------------------------------------------------------" >>${MY_ROOT_PATH}finish_history.log

scontrol show job $SLURM_JOB_ID     ### write job information to SLURM output file.
### js -j $SLURM_JOB_ID   ### write resource usage to SLURM output file (powertools command).
