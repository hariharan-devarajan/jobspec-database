#!/bin/bash -l
#SBATCH --job-name=pyt-profiler
#SBATCH --account=nn9987k
#SBATCH --time=00:10:00
#SBATCH --partition=accel
#SBATCH --nodes=1         #nbr of nodes
#SBATCH --ntasks=1        #nbr of tasks
#SBATCH --cpus-per-task=1 #nbr of threads (or cores)
#SBATCH --gpus=1          #total nbr of gpus
#SBATCH --gpus-per-node=1 #nbr of gpus per node
#SBATCH --mem=4G          #main memory
#SBATCH -o PyTprofiler.out

#define paths
Mydir=/cluster/projects/nn9987k/PyTorchProfiler
MyContainer=${Mydir}/Container/pytorch_22.12-py3.sif
MyExp=${Mydir}/examples

#specify bind paths by setting the environment variable
#export SINGULARITY_BIND="${MyExp},$PWD"

#TF32 is enabled by default in the NVIDIA NGC TensorFlow and PyTorch containers 
#To disable TF32 set the environment variable to 0
#export NVIDIA_TF32_OVERRIDE=0

#to run singularity container 
singularity exec --nv -B ${MyExp} ${MyContainer} python3 ${MyExp}/resnet18_profiler_api_4batch.py

echo 
echo "--Job ID:" $SLURM_JOB_ID
echo "--total nbr of gpus" $SLURM_GPUS
echo "--nbr of gpus_per_node" $SLURM_GPUS_PER_NODE

