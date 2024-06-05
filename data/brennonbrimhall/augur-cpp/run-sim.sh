#!/bin/bash

# This script was originally intended to submit augur test jobs to BYU's
# supercomputing laboratory.  I've redacted things that are spectific to BYU.  
# It may be useful to others, which is why we're including it here.  This should
# work for any facility that uses Slurm as the scheduler.

#SBATCH --time=01:00:00		# walltime 
#SBATCH --ntasks=24		# number of processor cores (i.e. tasks) 
#SBATCH --nodes=1		# number of nodes 
#SBATCH --mem-per-cpu=1024M	# memory per CPU core 
#SBATCH -J "AUGUR_TEST"		# job name 
#SBATCH --mail-user=  		# put your email address here
#SBATCH --mail-type=BEGIN 
#SBATCH --mail-type=END 
#SBATCH --mail-type=FAIL 
#SBATCH --qos=test

# Set the max number of threads to use for programs using OpenMP. Should be <=
# ppn. This will only modify augur's behavior if it is build with OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Actually run simulations here.  Two examples are given.  The first is a full
# simulation, the second is a Monte Carlo approximation.

# /full/path/to/augur/executable \
# 	/full/path/to/rankings \
#	/full/path/to/schedule \
#	rankingTreeDepth \
#	> out.txt

# /full/path/to/augur/executable \
# 	/full/path/to/rankings \
#	/full/path/to/schedule \
#	rankingTreeDepth \
#	numSims \
#	> out.txt

