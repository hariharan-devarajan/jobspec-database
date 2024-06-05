#!/bin/bash
#SBATCH --chdir /home/ramdass/2D-Human-Pose-Estimation/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 150G
#SBATCH --partition gpu
#SBATCH --time 20:00:00
#SBATCH --gres gpu:2
#SBATCH --qos dlav
#SBATCH --account civil-459-2023


./OmniPose/run_train.sh