#!/bin/bash -eu
#PBS -l select=1:ncpus=4
#PBS -l walltime=24:00:00

# Parameters
software_kind=apl # util or apl
name=charmm
version=c47b2
module_loaded="gcc-toolset/11.4.0"

PREFIX=$PREFIX_APP_DIR/packages/$software_kind/$name/$version

module load $module_loaded

cd /lwork/users/$USER/$PBS_JOBID

# Download source
cp $APPLICATION_REPOSITORY_PATH/$name/$version/$version.tar.gz .

# Build
## charmm requires source directory exists
source_path=$PREFIX_APP_DIR/sources/$software_kind/$name/$version
tar -xvf $version.tar.gz
mkdir -p `dirname $source_path`
mv charmm $source_path
cd $source_path

./configure --prefix $PREFIX
make -C build/cmake -j `nproc`
make -C build/cmake install

## Ensure all users have executable permission for all directories
find $PREFIX_APP_DIR/packages/$software_kind/$name/ -type d -exec chmod 755 {} \;

# Create modulefile
module_file_path=$PREFIX_APP_DIR/modules/$software_kind/$name/$version
mkdir -p `dirname $module_file_path`
# Set template variables
# Commands below assume that TEMPLATE_PREFIX is used in template_modulefile
export TEMPLATE_PREFIX=$PREFIX
export TEMPLATE_DEPENDS_ON=$module_loaded
envsubst < $APPLICATION_REPOSITORY_PATH/$name/template_modulefile > $module_file_path

## Ensure all users have executable permission for all directories
## and read permission for all files
find $PREFIX_APP_DIR/modules/$software_kind/$name -type d -exec chmod 755 {} \;
chmod 644 $PREFIX_APP_DIR/modules/$software_kind/$name/$version
