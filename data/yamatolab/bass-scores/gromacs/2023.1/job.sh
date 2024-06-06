#!/bin/bash -eu
#PBS -l select=1:ncpus=12:mpiprocs=8
#PBS -l walltime=03:00:00

# The test suite of gromacs requires 8 MPI processes.

# Implicit parameters
#  - PREFIX_APP_DIR (e.g. /apl)
#  - APPLICATION_REPOSITORY_PATH (e.g. /home/users/apl_manager/pbs-cluster-applications)

# Explicit parameters
software_kind=apl # or util
name=gromacs
version=2023.1
module_loaded="openmpi/4.1.5"

module load $module_loaded

PREFIX=$PREFIX_APP_DIR/packages/$software_kind/$name/$version

WORK_DIR=/lwork/users/$USER/$PBS_JOBID
cd $WORK_DIR

# Download source
wget "https://ftp.gromacs.org/gromacs/gromacs-${version}.tar.gz"
tar -xvf "gromacs-${version}.tar.gz"
cd "gromacs-${version}"

# Build

## Normal version
mkdir build
cd build
cmake .. \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DREGRESSIONTEST_DOWNLOAD=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX
make -j `nproc`
# Unset OMP_NUM_THREADS to avoid errors in test.
# OMP_NUM_THREADS is set by default in PBS.
unset OMP_NUM_THREADS
make check
make install

## MPI version
rm -rf *
cmake .. \
    -DGMX_BUILD_OWN_FFTW=ON \
    -DREGRESSIONTEST_DOWNLOAD=ON \
    -DGMX_MPI=ON \
    -DCMAKE_INSTALL_PREFIX=$PREFIX
make -j `nproc`
# Unset OMP_NUM_THREADS to avoid errors in test.
# OMP_NUM_THREADS is set by default in PBS.
unset OMP_NUM_THREADS
make check
make install

# By default, the installation directory is not executable by group.
find $PREFIX_APP_DIR/packages/$software_kind/$name/ -type d -exec chmod 755 {} \;

# Create modulefile
module_file_path=$PREFIX_APP_DIR/modules/$software_kind/$name/$version
mkdir -p `dirname $module_file_path`
# Set template variables
export TEMPLATE_PREFIX=$PREFIX
export TEMPLATE_DEPENDS_ON=$module_loaded
envsubst < $APPLICATION_REPOSITORY_PATH/$name/template_modulefile > $module_file_path

## Ensure all users have executable permission for all directories
## and read permission for all files
find $PREFIX_APP_DIR/modules/$software_kind/$name -type d -exec chmod 755 {} \;
chmod 644 $PREFIX_APP_DIR/modules/$software_kind/$name/$version
