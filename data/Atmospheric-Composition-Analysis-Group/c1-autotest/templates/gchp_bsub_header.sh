#!/usr/bin/bash
#BSUB -q ${JOB_QUEUE}
#BSUB -n ${NUM_CORES}
#BSUB -W 336:00
#BSUB -R "rusage[mem=300000] span[ptile=${CORES_PER_NODE}] select[mem < 2000000]"
#BSUB -a 'docker(${CONTAINER})'
#BSUB -J "Test"
#BSUB -g /liam.bindle/small_jobs
#BSUB -o lsf-run-%J.txt

# Source bashrc
. /etc/bashrc

# Set up runtime environment
set -x                           # Print executed commands
set -e                           # Exit immediately if a command fails
ulimit -c 0                      # coredumpsize
ulimit -l unlimited              # memorylocked
ulimit -u 50000                  # maxproc
ulimit -v unlimited              # vmemoryuse
ulimit -s unlimited              # stacksize
