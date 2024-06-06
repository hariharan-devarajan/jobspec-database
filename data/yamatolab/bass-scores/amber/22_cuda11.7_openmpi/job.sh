#!/bin/bash -eu
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=24:00:00

# Parameters
software_kind=apl
name=amber
cuda_version=11.7
openmpi_version=4.1.6
version="22_cuda${cuda_version}_openmpi"

PREFIX=$PREFIX_APP_DIR/packages/$software_kind/$name/$version

WORK_DIR=/lwork/users/$USER/$PBS_JOBID
cd $WORK_DIR

# Download source
cp $APPLICATION_REPOSITORY_PATH/src/Amber22.tar.bz2 .
cp $APPLICATION_REPOSITORY_PATH/src/AmberTools23.tar.bz2 .

# Build
tar -xvf Amber22.tar.bz2
tar -xvf AmberTools23.tar.bz2
cd amber22_src/

# Apply official patches
module load python
echo 'y' | ./update_amber --upgrade
./update_amber --update
module unload python

module purge
module load cuda/11.7
module load openmpi/4.1.6

cd build
cmake ../ \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCOMPILER=GNU  \
    -DMPI=TRUE -DCUDA=TRUE -DINSTALL_TESTS=TRUE \
    -DBUILD_GUI=FALSE \
    -DDOWNLOAD_MINICONDA=TRUE \
    2>&1 | tee cmake.log
make -j `nproc` 2>&1 | tee make.log
make install 2>&1 | tee make_install.log

mkdir -p $PREFIX/build_logs
cp cmake.log make.log make_install.log $PREFIX/build_logs

# Create modulefile
module_file_path=$PREFIX_APP_DIR/modules/$software_kind/$name/$version
mkdir -p `dirname $module_file_path`
# Set template variables
# Commands below assume that TEMPLATE_PREFIX is used in template_modulefile
export TEMPLATE_PREFIX=$PREFIX
export TEMPLATE_CUDA_VERSION=$cuda_version
export TEMPLATE_OPENMPI_VERSION=$openmpi_version
envsubst '$TEMPLATE_PREFIX $TEMPLATE_CUDA_VERSION $TEMPLATE_OPENMPI_VERSION' \
   < $APPLICATION_REPOSITORY_PATH/$name/$version/template_modulefile \
   > $module_file_path

## Ensure all users have executable permission for all directories
## and read permission for all files
find $PREFIX_APP_DIR/packages/$software_kind/$name -type d -exec chmod 755 {} \;

find $PREFIX_APP_DIR/modules/$software_kind/$name -type d -exec chmod 755 {} \;
chmod 644 $PREFIX_APP_DIR/modules/$software_kind/$name/$version
