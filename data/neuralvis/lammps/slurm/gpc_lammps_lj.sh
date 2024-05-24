#!/bin/bash --login

###
# job name
#SBATCH --job-name=E0001
# specify its partition
#SBATCH --partition=workq
# job stdout file
#SBATCH --output=E0001.%J.out
# job stderr file
#SBATCH --error=E0001.%J.err
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
export LAMMPS_NC=50
export LAMMPS_PPN=4
export GPCNET_NC=50
export GPCNET_PPN=64

# Define directories and files
export APP_BASE_DIR=/home/users/msrinivasa/develop
export DATA_DIR=./data
export EXPERIMENT_METAFILE=$EXPERIMENT_NAME.README.txt
export EXPERIMENT_JOBFILE=$EXPERIMENT_NAME.JOBFILE.csv
mkdir -p $DATA_DIR

# Write metadata into a README file for the experiment
echo $EXPERIMENT_NAME>$EXPERIMENT_METAFILE
echo "Total Allocation: "$TOTAL_NC>>$EXPERIMENT_METAFILE
echo "GPCNet Allocation: "$GPCNET_NC>>$EXPERIMENT_METAFILE
echo "LAMMPS Allocation: "$LAMMPS_NC>>$EXPERIMENT_METAFILE
echo "Nodelist: "$SLURM_JOB_NODELIST>>$EXPERIMENT_METAFILE
# Define a few craypat related environment variables for this jobstep
export PAT_RT_EXPDIR_BASE=$DATA_DIR/lammps/idle
# export PAT_RT_EXPFILE_MAX=$LAMMPS_PE_COUNT
mkdir -p $PAT_RT_EXPDIR_BASE

# Record the job start time
export LAMMPS_IDLE_START=`date -uI'seconds'`
# Run lammps without congestion with 4 ppn
srun --relative=0 \
     --nodes=$LAMMPS_NC \
     --ntasks-per-node=$LAMMPS_PPN \
    $APP_BASE_DIR/lammps/build/lmp+tracing \
    -i $APP_BASE_DIR/lammps/examples/DIFFUSE/in.msd.2d \
    > $PAT_RT_EXPDIR_BASE/lammps.idle.out
# Record the job end time
export LAMMPS_IDLE_END=`date -uI'seconds'`

# wait for 30 seconds
sleep 30

# Define a few craypat related environment variables for this jobstep
export PAT_RT_EXPDIR_BASE=$DATA_DIR/gpcnet
# export PAT_RT_EXPFILE_MAX=$LAMMPS_PE_COUNT
mkdir -p $PAT_RT_EXPDIR_BASE

# Record the job start time
export GPCNET_START=`date -uI'seconds'`
# Run gpcnet on its allocation with 64 ppn
srun --relative=$LAMMPS_NC \
     --nodes=$GPCNET_NC \
     --ntasks-per-node $GPCNET_PPN \
     $APP_BASE_DIR/GPCNET/network_load_test+pat \
     > $PAT_RT_EXPDIR_BASE/gpcnet.out &
# wait for a minute till gpcnet primes up the network
sleep 30

# Define a few craypat related environment variables for this next jobstep
export PAT_RT_EXPDIR_BASE=$DATA_DIR/lammps/congested
# export PAT_RT_EXPFILE_MAX=$LAMMPS_PE_COUNT
mkdir -p $PAT_RT_EXPDIR_BASE

# Record the job start time
export LAMMPS_CONGESTED_START=`date -uI'seconds'`
# Run lammps with congestion with 4 ppn
srun --relative=0 \
     --nodes=$LAMMPS_NC \
     --ntasks-per-node=$LAMMPS_PPN \
     $APP_BASE_DIR/lammps/build/lmp+tracing \
     -i $APP_BASE_DIR/lammps/examples/DIFFUSE/in.msd.2d \
     > $PAT_RT_EXPDIR_BASE/lammps.congested.out
# Record the job end time
export LAMMPS_CONGESTED_END=`date -uI'seconds'`

# wait till all jobsteps finish
wait

# now we know GPCNET is done
export GPCNET_END=`date -uI'seconds'`

# record all jobsteps
echo "start_time,end_time,job_id,job_name,user">$EXPERIMENT_JOBFILE
echo $LAMMPS_IDLE_START,$LAMMPS_IDLE_END,$EXPERIMENT_NAME.01,$EXPERIMENT_NAME.LAMMPS.IDLE,$USER>>$EXPERIMENT_JOBFILE
echo $GPCNET_START,$GPCNET_END,$EXPERIMENT_NAME.02,$EXPERIMENT_NAME.GPCNET.AGGRESSOR,$USER>>$EXPERIMENT_JOBFILE
echo $LAMMPS_CONGESTED_START,$LAMMPS_CONGESTED_END,$EXPERIMENT_NAME.03,$EXPERIMENT_NAME.LAMMPS.CONGESTED,$USER>>$EXPERIMENT_JOBFILE


# Clean up GPCNET output, since we don't need it
rm *.dat
rm *.rec

# Clean up the lammps log output (it is redundant)
rm log.lammps
