#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=res17                            # sets the job name if not set from environment
#SBATCH --output cmllogs/%x_%j.log                   # redirect STDOUT to; %j is the jobid, _%j is array task id
#SBATCH --error cmllogs/%x_%j.log                    # redirect STDERR to; %j is the jobid,_%j is array task id
#SBATCH --account=scavenger                           # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger                                 # set QOS, this will determine what resources can be requested
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenger
#SBATCH --mem 16gb                                      # memory required by job; MB will be assumed
#SBATCH --mail-user arjung15@umd.edu
#SBATCH --mail-type=END,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --time=10:00:00                                 # how long will the job will take to complete; format=hh:mm:ss

python generate_poison.py cfg_CIFAR/singlesource_singletarget_binary_finetune_3/experiment_0017.cfg &&
python finetune_and_test.py cfg_CIFAR/singlesource_singletarget_binary_finetune_3/experiment_0017.cfg