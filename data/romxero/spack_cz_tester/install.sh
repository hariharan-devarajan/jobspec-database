#!/bin/bash
#set -x #debug 
#SBATCH --job-name=SpackBuilds
#SBATCH --output=SpackBuilds_%A_%a.out
#SBATCH --error=SpackBuilds_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1 # how to scale the following to multiple tasks?
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G


if [ -z "${SLURM_CPUS_PER_TASK}" ]
then 
	CORECOUNT=4
else
	CORECOUNT=${SLURM_CPUS_PER_TASK} #grab the core count from slurm task
fi

#global variables
#CORECOUNT=${SLURM_CPUS_PER_TASK} #grab the core count from slurm task
ARCH="x86_64" #this is the main target, select either x86_64, zen2, or skylake_avx512
BASE_GCC_VERS=5.5.0 #the version of gcc that will be used to build other variations of gcc 

#the compilers we will need.
compilers=(
    %gcc@13.1.0
)

mpis=(
    openmpi
    mpich
)

#clone the spack repo into this current directory
git clone -c feature.manyFiles=true https://github.com/spack/spack.git

#backup old yaml files
mv spack/etc/spack/defaults/packages.yaml spack/etc/spack/defaults/packages.yaml_bak
mv spack/etc/spack/defaults/modules.yaml spack/etc/spack/defaults/modules.yaml_bak

#copy over the configuration files:
cp defaults/modules.yaml spack/etc/spack/defaults/modules.yaml
cp defaults/packages.yaml spack/etc/spack/defaults/packages.yaml

#source the spack environment
source spack/share/spack/setup-env.sh

#install compilers 
#fix was added due to zen2 not having optimizations w/ 4.8.5 compiler
spack install -j${CORECOUNT} gcc@${BASE_GCC_VERS} target=x86_64


#now add the compilers - gcc
spack compiler find `spack location --install-dir gcc@${BASE_GCC_VERS}`
spack compiler find `spack location --install-dir gcc@${BASE_GCC_VERS}`/bin


#############SOFTWARE INSTALL########################

for compiler in "${compilers[@]}"
do
    # Serial installs
    spack install -j${CORECOUNT} proj $compiler target=${ARCH}
    spack install -j${CORECOUNT} swig $compiler target=${ARCH}
    spack install -j${CORECOUNT} maven $compiler target=${ARCH}
    spack install -j${CORECOUNT} geos $compiler target=${ARCH}

    # Parallel installs
    for mpi in "${mpis[@]}"
    do
        spack install -j${CORECOUNT} $mpi $compiler target=${ARCH}
        spack install -j${CORECOUNT} cdo  $compiler ^$mpi target=${ARCH}
        spack install -j${CORECOUNT} parallel-netcdf $compiler ^$mpi target=${ARCH}
        spack install -j${CORECOUNT} petsc $compiler ^$mpi target=${ARCH}
		spack install -j${CORECOUNT} netcdf-fortran $compiler ^$mpi target=${ARCH}
		spack install -j${CORECOUNT} netcdf-c $compiler ^$mpi target=${ARCH}
		spack install -j${CORECOUNT} netcdf-cxx4 $compiler ^$mpi target=${ARCH}
		spack install -j${CORECOUNT} hdf5 $compiler ^$mpi target=${ARCH}
		spack install -j${CORECOUNT} fftw $compiler ^$mpi target=${ARCH}
		spack install -j${CORECOUNT} parallelio $compiler ^$mpi target=${ARCH}
		spack install -j${CORECOUNT} dealii $compiler ^$mpi target=${ARCH}
		spack install -j${CORECOUNT} cgal $compiler ^$mpi target=${ARCH}
    done
done


#################END_OF_SOFTWARE_INSTALLS####################################


#have spack regenerate module files:

spack module lmod refresh --delete-tree -y


exit 0

