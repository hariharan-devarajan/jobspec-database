##!/bin/bash
#BSUB -P csc330
#BSUB -W 00:30
##BSUB -w ended(######)
#BSUB -nnodes 8
#BSUB -alloc_flags "nvme smt4"
#BSUB -J DeepCam_profile
#BSUB -o %J.out
#BSUB -e %J.err
## End LSF directives and begin shell commands

# Run parameters
export BATCHSIZE=2
export DO_PROFILING='false' # true or false
export DO_NCCL_DEBUG='true' # true or false

# Setup software environment
#module load cuda/10.2.89
source ~/.mlperf_deepcam_profile
module unload python
conda activate open-ce-0.1-0 # has basemap
module unload darshan
#module load pytorch/v1.6.0-gpu # TODO
#module load gcc                # TODO
#module load mpich              # TODO
#module load nccl/2.5.6         # TODO

# Setup software environment
export PYTHONPATH=$PYTHONPATH:/ccs/home/mrowan/code/mlperf-logging # Add as part of directory?
export NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
#export BASEMAPDATA=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
#export PROJ_LIB=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/share/basemap
#export PYTHONPATH=/global/homes/m/mrowan/.conda/envs/mlperf_deepcam/lib/python3.7/site-packages:/global/homes/m/mrowan/code/mlperf-logging

# XLA environment
#source $WORLDWORK/stf011/junqi/native-build/latest/1.14.0/env.sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_DIR
export PYTHONPATH=$(pwd):$PYTHONPATH # TODO

if [ "$DO_PROFILING" == "true" ]
then
    module load nsight-systems
fi

# Copy data to burst buffer # TODO
export data_dir_prefix="/ccs/home/mrowan/scratch/ml-performance-benchmark/deepcam/summit/data/cam5_data/All-Hist_small_split_${NODES}"
#jsrun -n$NODES -a1 -c42 -r1 cp -rL ./data/cam5_data/All-Hist_small_split_${NODES}/ /mnt/bb/$USER
#export data_dir_prefix=/mnt/bb/$USER

if [ "$DO_PROFILING" == "true" ]
then
    jsrun -n$((NODES*6)) -a1 -c7 -g1 -r6 \
          --bind=proportional-packed:7 \
          -x LD_LIBRARY_PATH \
          stdbuf -o0 \
          ./utils/run_with_profiling.sh
else
    jsrun -n$((NODES*6)) -a1 -c7 -g1 -r6 \
          --bind=proportional-packed:7 \
          -x LD_LIBRARY_PATH \
          stdbuf -o0 \
          ./utils/run.sh
fi

# if [ "$DO_PROFILING" == "true" ]
# then
#     cat ./utils/run_with_profiling.sh >> log.${LSB_JOBID}
# else
#     cat ./utils/run.sh >> log.${LSB_JOBID}
# fi

# mv log.${LSB_JOBID} $LOG_DIR/
# mv %J.out $LOG_DIR/
# mv %J.err $LOG_DIR/


