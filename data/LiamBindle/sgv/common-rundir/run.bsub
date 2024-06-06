#!/usr/bin/env bash
#BSUB -q general
#BSUB -n NUM_CORES
#BSUB -W 24:00
#BSUB -R "rusage[mem=300000] span[ptile=CORES_PER_NODE]"
#BSUB -a 'docker(registry.gsc.wustl.edu/sleong/base-engineering-gcc)'
#BSUB -J "SGV run"
#BSUB -g /$USER/benchmarking
#BSUB -N
#BSUB -u liam.bindle@wustl.edu
#BSUB -o lsf-run-%J-output.txt

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

# Execute simulation
cd COMPUTE_NODE_RUNDIR
rm -f cap_restart
rm -f OutputDir/*
./runConfig.sh
export TMPDIR="$__LSF_JOB_TMPDIR__"
mpirun -x LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH -np NUM_CORES ./geos
