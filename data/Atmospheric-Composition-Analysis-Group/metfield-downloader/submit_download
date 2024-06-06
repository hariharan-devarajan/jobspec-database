#!/bin/bash
set -e
set -u

# Source CONF_FILE and make sure WORKINGDIR and WGET_OPTIONS are defined
CONF_FILE=$1
[ "$CONF_FILE" = "-" ] || source $CONF_FILE
: "$WORKINGDIR" "$WGET_OPTIONS" "$NAME"

# Load WGET_URLS from file or stdin (-)
WGET_URLS=$(cat $2)

WORKINGDIR=$(realpath $WORKINGDIR)

JOB_RPATH="$(date +"%Y-%m-%d")/$RANDOM"
DOWNLOAD_CTRL_DIR=/storage1/fs1/rvmartin/Active/GEOS-Chem-shared/MetFieldProcessing/downloads/jobs
JOB_PATH=$DOWNLOAD_CTRL_DIR/$JOB_RPATH

# Make job control directory
mkdir -p $JOB_PATH

date > $JOB_PATH/date.txt

echo "$WGET_URLS" > $JOB_PATH/urls.txt

echo """WORKINGDIR=$WORKINGDIR
WGET_OPTIONS='$WGET_OPTIONS'""" > $JOB_PATH/conf.rc

echo """#!/bin/sh
#BSUB -n 2
#BSUB -R 'rusage[mem=8000] span[hosts=1]'
#BSUB -W 168:00
#BSUB -q general
#BSUB -g /liam.bindle/downloads
#BSUB -a 'docker(inutano/wget)'
#BSUB -J "Download$NAME"
#BSUB -Ne
#BSUB -u liam.bindle@wustl.edu
#BSUB -o $JOB_PATH/output.txt

set -e

mkdir -p $WORKINGDIR
cd $WORKINGDIR
wget $WGET_OPTIONS -i $JOB_PATH/urls.txt""" > $JOB_PATH/wget.bsub

echo "Created download: $JOB_PATH"
bsub < $JOB_PATH/wget.bsub
