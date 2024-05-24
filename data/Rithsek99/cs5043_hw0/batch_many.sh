#!/bin/bash
# Andrew H. Fagg
#
# Example with an array of experiments
#  The --array line says that we will execute 4 experiments (numbered 0,1,2,3).
#   You can specify ranges or comma-separated lists on this line
#  For each experiment, the SLURM_ARRAY_TASK_ID will be set to the experiment number
#   In this case, this ID is used to set the name of the stdout/stderr file names
#   and is passed as an argument to the python program
#
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal
#

#SBATCH --partition=debug_5min
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=1024
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=results/hw0_exp%04a_stdout.txt
#SBATCH --error=results/hw0_exp%04a_stderr.txt
#SBATCH --time=00:02:00
#SBATCH --job-name=hw0_test
#SBATCH --mail-user=rithsek.ngem-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504312/cs5043/hw/hw_0
#SBATCH --array=0-9
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up

. /home/fagg/tf_setup.sh
conda activate tf

# Change this line to start an instance of your experiment
python hw_0.py --epochs 7000 --hidden 16 --exp $SLURM_ARRAY_TASK_ID

