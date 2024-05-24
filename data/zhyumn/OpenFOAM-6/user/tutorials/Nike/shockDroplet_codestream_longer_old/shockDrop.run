#!/bin/bash
# HEADER for Parallel job using 80 processors:
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks-per-node=128    # number of processors per node
#SBATCH --cpus-per-task=1       # number of cpus per task
#SBATCH -t 148:00:00      # run for 3 hr max
#SBATCH --partition=ROME
###SBATCH --mail-type=begin   # send email when process begins...
###SBATCH --mail-type=end     # ...and when it ends...
###SBATCH --mail-type=fail    # ...or when it fails.
###SBATCH --mail-user=<your-email>@princeton.edu # send notifications to this email
#SBATCH -e job.err              # Name of output file for error messages
#SBATCH -o job.out              # Name of output file for standard output



# BODY - commands to be run
# Load required modules
. /home/zhan6305/OpenFOAM/cleanOpenFOAM/OpenFOAM-6/etc/bashrc
. /home/zhan6305/OpenFOAM/cleanOpenFOAM/OpenFOAM-6/user/bashrc
. /home/zhan6305/OpenFOAM/cleanOpenFOAM/OpenFOAM-6/user/etc/bashrc

blockMesh
cp -r 0.orig 0
decomposePar -latestTime -force
mpirun -np 128 rhoCentralRealgasFoam  -parallel
reconstructPar -latestTime
rm -r 0
mv  0.0005 0
setFields
decomposePar -latestTime   -force

sed -i 's/endTime         0.0005;/endTime         0.005;/' system/controlDict 
mpirun -np 128 rhoCentralRealgasFoam  -parallel
reconstructPar -newTimes
