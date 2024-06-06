#!/usr/bin/bash
#BSUB -q rvmartin
#BSUB -n 60
#BSUB -W 192:00
#BSUB -R "rusage[mem=300000] span[ptile=30] select[mem < 2000000]"
#BSUB -a 'docker(registry.gsc.wustl.edu/sleong/base-engineering-gcc)'
#BSUB -J "1yr-restart"
#BSUB -g /$USER/E2
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
cd /scratch1/C360/2020-04/scratch-1
chmod +x runConfig.sh geos
./runConfig.sh
export TMPDIR="$__LSF_JOB_TMPDIR__"
mpirun -x LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH -np 60 ./geos
