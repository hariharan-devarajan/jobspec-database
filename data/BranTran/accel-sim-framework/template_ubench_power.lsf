#!/bin/bash
#BSUB -P gen150
#BSUB -W 2:00
#BSUB -J AccelWattch_ubench
#BSUB -nnodes 1
#BSUB -N tranbq@ornl.gov
#Run this script from the root of the accelwattch artifact directory

#Needed modules
module load gcc/7.5.0 #7.5.0 for summit, 6.5.0 if Andes 
module load cuda/11.0.2
module load makedepend
module load python/2.7.15
module load cmake

#Get host/GPU information
echo "####START INFO####"
echo "Host information: Node"
echo $LSB_MCPU_HOST
echo $LSB_HOSTS | tr ' ' '\n' | sort | grep -v 'batch' | uniq
echo "GPU Information: Device_ID GPU_ID"
nvidia-smi -L | awk '{print NR-1, $NF}' | tr -d '[)]'
echo "####END   INFO####"

#Setting up environment variables
export CUDA_INSTALL_PATH=$(dirname "$(dirname `which nvcc`)")
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

#Move to the correct directory
cd $MEMBERWORK/gen150/accelwattch-artifact-appendix
pwd

#Setup environment variables with source
source ./accel-sim-framework/gpu-simulator/setup_environment.sh
echo $ACCELSIM_ROOT

#export CUDA_VISIBLE_DEVICES=<GPU_DEVID>
jsrun -n1 -c1 -g1 $ACCELSIM_ROOT/../accelwattch_hw_profiler/profile_ubench_power.sh 0 REPLACE_TEXT &
jsrun -n1 -c1 -g1 $ACCELSIM_ROOT/../accelwattch_hw_profiler/profile_ubench_power.sh 0 REPLACE_TEXT &
jsrun -n1 -c1 -g1 $ACCELSIM_ROOT/../accelwattch_hw_profiler/profile_ubench_power.sh 0 REPLACE_TEXT &
jsrun -n1 -c1 -g1 $ACCELSIM_ROOT/../accelwattch_hw_profiler/profile_ubench_power.sh 0 REPLACE_TEXT &
jsrun -n1 -c1 -g1 $ACCELSIM_ROOT/../accelwattch_hw_profiler/profile_ubench_power.sh 0 REPLACE_TEXT &
jsrun -n1 -c1 -g1 $ACCELSIM_ROOT/../accelwattch_hw_profiler/profile_ubench_power.sh 0 REPLACE_TEXT &
wait

