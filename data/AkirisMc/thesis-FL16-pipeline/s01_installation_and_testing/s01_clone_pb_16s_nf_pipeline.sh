#!/bin/bash

# Clone pb_16s_nf pipeline from GitHub
# Akiris Moctezuma  Jun 2023
# See https://github.com/PacificBiosciences/pb-16S-nf

# Note: this script should be run on a compute node
# qsub s01_clone_pb_16s_nf_pipeline.sh

# PBS directives
#---------------

#PBS -N s01_clone_pb_16s_nf_pipeline
#PBS -l nodes=1:ncpus=4
#PBS -l walltime=00:30:00
#PBS -q half_hour 
#PBS -m abe 
#PBS -M akiris.moc.901@cranfield.ac.uk 

#===============
#PBS -j oe
#PBS -W sandbox=PRIVATE
#PBS -k n
ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID
## Change to working directory
cd $PBS_O_WORKDIR
## Calculate number of CPUs
export cpus=`cat $PBS_NODEFILE | wc -l`
## =============

# Start message
echo "Started"
date
echo ""

# Stop at runtime errors
set -e

# Load git module
module load git/2.23.0-GCCcore-8.3.0-nodocs
git --version
echo ""

# Create pipeline directory in tools directory
# (later conda version of NF will create an additional nf_conda directory in the user home directory for singularity files)
project_folder="/scratch/s394901/thesis"
mkdir -p "${project_folder}"/tools
cd "${project_folder}"/tools

# Download the pipeline
git clone https://github.com/PacificBiosciences/pb-16S-nf.git

# Completion message
echo ""
echo "Done"
date

# --- Your code ends here --- #

## Tidy up the log directory
## =========================
rm $PBS_O_WORKDIR/$PBS_JOBID
