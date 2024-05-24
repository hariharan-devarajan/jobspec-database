#!/bin/bash --login

###
# job name
#SBATCH --job-name=E0002
# specify its partition
#SBATCH --partition=workq
# job stdout file
#SBATCH --output=E0002.%J.out
# job stderr file
#SBATCH --error=E0002.%J.err
# maximum job time in HH:MM:SS
#SBATCH --time=01:00:00
#SBATCH --nodes=100
# maximum memory
#SBATCH --mem-per-cpu=512M
# run a single task
###

module restore PrgEnv-cray
module load perftools-base
module load perftools

# Define experiment name
export EXPERIMENT_NAME=$SLURM_JOB_NAME

# Define allocations
export TOTAL_NC=$SLURM_JOB_NUM_NODES
export LAMMPS_NC=100
export LAMMPS_PPN=64
# Define scaling for LAMMPS lj problem
# See https://github.com/neuralvis/lammps/tree/master/bench
# for details
export Px=15
export Py=15
export Pz=15

# Define directories and files
export APP_BASE_DIR=/lus/cls01053/msrinivasa/develop
export LMP_ROOT=$APP_BASE_DIR/lammps
export DATA_DIR=./data
export EXPERIMENT_METAFILE=$EXPERIMENT_NAME.README.txt
export EXPERIMENT_JOBFILE=$EXPERIMENT_NAME.JOBFILE.csv
mkdir -p $DATA_DIR

# Write metadata into a README file for the experiment
echo $EXPERIMENT_NAME>$EXPERIMENT_METAFILE
echo "Total Allocation: "$TOTAL_NC>>$EXPERIMENT_METAFILE
echo "LAMMPS Allocation: "$LAMMPS_NC>>$EXPERIMENT_METAFILE
echo "Nodelist: "$SLURM_JOB_NODELIST>>$EXPERIMENT_METAFILE
# Define a few craypat related environment variables for this jobstep
export PAT_RT_EXPDIR_BASE=$DATA_DIR/lammps/idle
# export PAT_RT_EXPFILE_MAX=$LAMMPS_PE_COUNT
mkdir -p $PAT_RT_EXPDIR_BASE

# Record the job start time
export LAMMPS_IDLE_START=`date -uI'seconds'`
# Run lammps without congestion with 64 ppn
srun --relative=0 \
     --nodes=$LAMMPS_NC \
     --ntasks-per-node=$LAMMPS_PPN \
     $LMP_ROOT/build/lmp+tracing \
     -var x $Px -var y $Py -var z $Pz \
     -in $LMP_ROOT/perftools/inputfiles/in.rhodo.scaled \
     > $PAT_RT_EXPDIR_BASE/lammps.idle.out
# Record the job end time
export LAMMPS_IDLE_END=`date -uI'seconds'`

# record all jobsteps
echo "start_time,end_time,job_id,job_name,user">$EXPERIMENT_JOBFILE
echo $LAMMPS_IDLE_START,$LAMMPS_IDLE_END,$EXPERIMENT_NAME.01,$EXPERIMENT_NAME.LAMMPS.IDLE,$USER>>$EXPERIMENT_JOBFILE
