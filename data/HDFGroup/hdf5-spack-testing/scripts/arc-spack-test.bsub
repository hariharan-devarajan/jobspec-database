#!/bin/bash -l

#COBALT -t 30 -n 1 -q arcticus_debug
cd /gpfs/jlse-fs0/users/<username>/arc-spack

module use /soft/modulefiles
module load cmake
module load openmpi/4.1.1-gcc

#module load gcc/11.1.0 ## loaded by previous load

export CC=mpicc
export FC=mpif90

. share/spack/setup-env.sh

#spack install --test=root hdf5
spack install --test=root hdf5-vol-external-passthrough
spack install --test=root hdf5-vol-external-passthrough ^hdf5@develop-1.13
spack install hdf5-vol-async ^hdf5@develop-1.13
spack install --test=root hdf5-vol-log@master-1.1
spack install adios2+hdf5

echo ""
echo "Find installed hdf5 and vol packages"
echo ""
spack find hdf5-vol-external-passthrough
spack find hdf5-vol-async
spack find hdf5-vol-log
spack find hdf5
spack find adios2

echo ""
echo "Finished script arc-test-spack.bsub"

