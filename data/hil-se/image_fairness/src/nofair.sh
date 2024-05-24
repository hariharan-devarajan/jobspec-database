#!/bin/bash -l
## NOTE the -l flag!

## Name of the job -You'll probably want to customize this
#SBATCH -J image_fairness_nofair

## Use the resources available on this account
#SBATCH -A loop

## Standard out and Standard Error output files
#SBATCH -o log/%J_%a.o
#SBATCH -e log/%J_%a.e

## Request 3 Days, 5 Hours, 5 Minutes, 3 Seconds run time MAX,
## anything over will be KILLED
#SBATCH -t 3-03:05:03

## Put in tier3 partition for testing small jobs, like this one
## But because our requested time is over 4 day, it won't run, so
## use any tier you have available
#SBATCH -p tier3

## Request 1 GPU for one task, note how you can put multiple commands
## on one line
#SBATCH --gres=gpu:1

## Job memory requirements in MB
#SBATCH --mem=32G

## Job script goes below this line

spack unload -a
## Load modules with spack
## Tensorflow
spack load /lklqe3u
## matplotlib
spack load /saj4vss
## pandas
spack load py-pandas
## Execute target code
python3 main.py nofair ${SLURM_ARRAY_TASK_ID}

## Submit this job with
## sbatch --array=1-30 nofair.sh
