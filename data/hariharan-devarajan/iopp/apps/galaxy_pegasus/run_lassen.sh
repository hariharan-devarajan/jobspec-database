#!/bin/bash
### LSF syntax
#BSUB -cwd /usr/workspace/iopp/software/iopp/apps/galaxy_pegasus 
#BSUB -nnodes 1         #number of nodes
#BSUB -W 2:00             #walltime in minutes
#BSUB -G asccasc           #account
#BSUB -J galaxy_pegasus   #name of job
#BSUB -q pbatch            #queue to use

CPWD=/usr/workspace/iopp/software/iopp/apps/galaxy_pegasus
source /usr/workspace/iopp/install_scripts/bin/iopp-init
source /usr/workspace/iopp/install_scripts/bin/spack-init
spack env activate -p galaxy-pegasus-conda
export CONDA_ROOT=/usr/WS2/iopp/software/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/anaconda3-2021.05-wusktxlhsqaxq2xlxktr62dbswezotv3
source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate galaxy-pegasus

source /usr/WS2/iopp/software/htcondor-V9.0.6/install/bin/condor.sh

echo "Loading environment - Done"
sleep 5

export PEGASUS_HOME=/usr/WS2/iopp/software/pegasus/install
export PATH=$PEGASUS_HOME/bin:$PEGASUS_HOME/sbin:$PATH
/usr/WS2/iopp/software/htcondor-V9.0.6/install/sbin/condor_master

condor_status --long | grep GPU
echo "Loading condor - Done"

JOB_DIR=/p/gpfs1/iopp/temp/galaxy-pegasus-$USER/app
SCRATCH_DIR=/p/gpfs1/iopp/temp/galaxy-pegasus-$USER/scratch

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

echo "Running workflow"
pushd $JOB_DIR/run_dir
pegasus-remove $PWD
rm monitord.pid
pegasus-run $PWD
sleep $((2*3550))
pegasus-status -l $PWD
pegasus-remove $PWD

popd
popd
