#!/usr/bin/env bash
#SBATCH --partition=short        ### Partition (like a queue in PBS)
#SBATCH --job-name=DM_2     ### Job Name
#SBATCH --output=slurm-%j-%x         ### File in which to store job output
#SBATCH --time=0-10       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Number of nodes needed for the job
#SBATCH --ntasks-per-node=14     ### Number of tasks to be launched per Node

ml easybuild intel/2017a Python/3.6.1

./pt_one2.py
