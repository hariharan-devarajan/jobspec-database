#!/bin/bash

####### Refer to: https://slurm.schedmd.com/mc_support.html

#SBATCH --job-name=xxx

##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=xxx@go.olemiss.edu

#SBATCH --time=30-00:00:00
#SBATCH --nodes=1

####### -B socket[:core[:thread]]
#SBATCH -B 1:8:1

#SBATCH --mem-per-cpu=500M

#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=8
#SBATCH --ntasks-per-core=1
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

####### The current slurm job file uses NVIDIA GPU MPS accelerator for fast LAMMPS runs;
#######
#######
####### set "-B *:*:1", "--cpus-per-task=1", and "--hint=nomultithread" to temporarily
####### disable hyperthreading.

####### dell5760:  1*{cpu(8c/16t)@2.6GHz} + 1*{gpu(32cu)};

####### As of 03/24, slurm has been updated to 23.02.07 version.
#######
#######
#######
#######

####### use 1*8 MPI (100% CPU) + 1 GPU for the job will
####### be the best for efficiency (Ref: 100%); <the current one>
#######
#######
#######
#######

####### Change the email address in "--mail-user" entry if need notifications.






export CUDA_MPS_PIPE_DIRECTORY=$SLURM_SUBMIT_DIR/nvidia-mps.$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=$SLURM_SUBMIT_DIR/nvidia-log.$SLURM_JOB_ID
nvidia-cuda-mps-control -d
sleep 1

mpiexec --bind-to core --map-by core -n 8 lmp_2Aug2023_update3_more_gcc_sfft_openmpi_cuda_mps -sf gpu -pk gpu 1 neigh yes newton on pair/only off split 1.0 -i input.lmps

sleep 1
echo quit | nvidia-cuda-mps-control
sleep 1
rm -r $SLURM_SUBMIT_DIR/nvidia-*


