#!/bin/bash

#specify the time limit for the cron job
#SBATCH --time=5:0:0

# use cron.log and append to it
#SBATCH --output=cron.log
#SBATCH --open-mode=append

# the account, partition, and qos should not be changed
#SBATCH --account=cron-account
#SBATCH --partition=cron
#SBATCH --qos=cron

# Specify a valid cron string for the schedule
# this is to be submitted everyday at 11:59 PM
SCHEDULE='59 23 * * *'


SSH_PREFIX="ssh -i /home/$(whoami)/.ssh"       # prefix to ssh keys
ROSS_EXP=/project/rossc/Orofacial/BiteForce/Data     # top-level destination directory


# Some data are to be transferred with compression, others without.
# 1. choose appropriate rsync command for each source system
# 2. specify appropriate destination sub-directory "$ROSS_EXP/<...>"


RIPPLE_DATA="labadmin@128.135.180.217:/cygdrive/c/Users/labadmin/Trellis/dataFiles"

# no compression
rsync -av -e "$SSH_PREFIX/id_behavrsync"  $RIPPLE_DATA  $ROSS_EXP/<...>

# with compression
rsync -avz -e "$SSH_PREFIX/id_behavrsync"  $RIPPLE_DATA  $ROSS_EXP/<...>


RT_MATLAB_MODELS="labadmin@128.135.180.236:/cygdrive/c/Models"

# no compression
rsync -av -e "$SSH_PREFIX/id_behavrsync"  $RT_MATLAB_MODELS  $ROSS_EXP/<...>

# with compression
rsync -avz -e "$SSH_PREFIX/id_behavrsync" $RT_MATLAB_MODELS  $ROSS_EXP/<...>


PROCAPTURE="labadmin@128.135.180.159"
C_DATA=/cygdrive/c/Data 
D_DATA=/cygdrive/d/Data 
E_DATA=/cygdrive/e/Data 
F_DATA=/cygdrive/f/Data 

# no compression
rsync -av -e "$SSH_PREFIX/id_behavrsync" $PROCAPTURE:$C_DATA :$D_DATA :$E_DATA :$F_DATA  $ROSS_EXP/<...>

# with compression
rsync -avz -e "$SSH_PREFIX/id_behavrsync" $PROCAPTURE:$C_DATA :$D_DATA :$E_DATA :$F_DATA  $ROSS_EXP/<...> 


# resubmit this script with --begin set to the next scheduled cron time
# next-cron-time is a script that parses a cron schedule string and returns
# the next execution time
sbatch --quiet --begin=$(next-cron-time "$SCHEDULE") cron_template.sbatch
