#!/bin/bash
### LSF syntax
#BSUB -nnodes 32          #number of nodes
#BSUB -W 02:00                 #walltime in minutes
#BSUB -G asccasc           #account
#BSUB -J cm1  #name of job
#BSUB -q pbatch             #queue to use
workflow_index=WINDEX
NUM_NODES=32

# Usage: bsub run_lassen_per_workflow.sh <workflow index> 
# 0 - config_files/cpm_RadConvEquil
# 1 - config_files/dns_RayleighBenard
# 2 - config_files/hurricane_3d
# 3 - config_files/hurricane_axisymmetric
# 4 - config_files/les_ConvBoundLayer
# 5 - config_files/les_ConvPBL_moisture
# 6 - config_files/les_HurrBoundLayer
# 7 - config_files/les_ShallowCu
# 8 - config_files/les_ShallowCuLand
# 9 - config_files/les_ShallowCuPrecip
# 10 - config_files/les_ShearBoundLayer
# 11 - config_files/les_StableBoundLayer
# 12 - config_files/les_StratoCuDrizzle
# 13 - config_files/les_StratoCuNoPrecip
# 14 - config_files/nh_mountain_waves
# 15 - config_files/scm_HurrBoundLayer
# 16 - config_files/scm_HurrBoundLayer_tqnudge
# 17 - config_files/sea_breeze
# 18 - config_files/squall_line
# 19 - config_files/supercell
CWD=/usr/workspace/iopp/software/iopp/apps/cm1r20.3/run1
mkdir -p $CWD
source /usr/workspace/iopp/install_scripts/bin/iopp-init
export run_dir=$pfs/temp/cm1r20.3.$LSB_JOBID
mkdir -p $run_dir
tasks_per_node=40


cm1_executable_path=/usr/workspace/iopp/applications/cm1r20.3/install/bin
unset RECORDER_NO_MPI
date=$(date '+%Y%m%d')
# export RECORDER_TRACES_DIR=/p/gpfs1/haridev/iopp/recorder_logs/recorder-$date/cm1_$USER_$LSB_JOBID

recorder_lib_path=/usr/workspace/iopp/software/Recorder/install/lib/librecorder.so
tracer_lib_path=$recorder_lib_path

cp -r $cm1_executable_path/* $run_dir


pushd $run_dir

count=0
for d in config_files/*; do
  [ -d "$d" ] || continue
  if [ $count == $workflow_index ]; then
     workflow_dir="$d"
     echo "$count - $workflow_dir"
     break
  fi
  ((++count))
done

cp cm1.exe $workflow_dir

sed -i -E "s/ ppnode *= *[0-9]+,/ ppnode       =      $tasks_per_node,/" $run_dir/$workflow_dir/namelist.input
cat $run_dir/$workflow_dir/namelist.input | grep ppnode
pushd $workflow_dir

lrun -N $NUM_NODES -T $tasks_per_node --env "LD_PRELOAD=$tracer_lib_path" ./cm1.exe  > $CWD/cm1_${workflow_index}_${LSB_JOBID}_recorder.log
popd
popd
