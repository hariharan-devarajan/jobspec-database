#!/bin/bash
#SBATCH --job-name=MOM6     # create a short name for your job
#SBATCH -A cimes2
#SBATCH --nodes=1                # node count
##SBATCH --nodelist=stellar-m01g5 # specify node
#SBATCH --ntasks-per-node=16     # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --mem=512000M
#SBATCH --gres=gpu:1             # use GPU
#SBATCH --gpu-mps
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cheng.zhang@princeton.edu

rm -rf *.nc
module purge
module load anaconda3/2021.5 intel/2021.1.2 openmpi/intel-2021.1/4.1.0 hdf5/intel-2021.1/1.10.6 netcdf/intel-19.1/hdf5-1.10.6/4.7.4 cudatoolkit/11.3 
source activate ~/torch_gpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64:/home/cz3321/torch_gpu/lib
# export CUDA_VISIBLE_DEVICES="0"
# nvidia-cuda-mps-control -d

srun -n 16 ../../../build/intel/ocean_only/repro/MOM6