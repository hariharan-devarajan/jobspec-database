#!/bin/bash
#SBATCH -n 1 
#SBATCH -t 24:00:00 

mkdir -p /ibex/user/$USER/singularity/tmpdir
export SINGULARITY_TMPDIR=/ibex/user/$USER/singularity/tmpdir
 
module load singularity

#build image
singularity build --fakeroot --force singularityexample.sig singularityexample.def

#run image
singularity run /home/ruizdeam/singularityexample.sig