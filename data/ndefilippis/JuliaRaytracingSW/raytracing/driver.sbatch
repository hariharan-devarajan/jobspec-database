#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=2L_ray

module purge

export NUM_JULIA_THREADS=`nproc`

rundir=$SCRATCH/twolayer_raytracing/$SLURM_JOB_ID
mkdir -p $rundir
cp driver.sbatch Driver.jl TwoLayerRaytracing.jl Raytracing.jl initial_condition.jld2 $rundir
cp initial_condition_U0.1.jld2 $rundir/initial_condition.jld2
cp Parameters.jl $rundir/Parameters.jl
cd $rundir

#~/julia-1.9.4/julia-writable -t $SLURM_CPUS_PER_TASK Driver.jl > run.log
julia -t $SLURM_CPUS_PER_TASK Driver.jl > run.log

exit
