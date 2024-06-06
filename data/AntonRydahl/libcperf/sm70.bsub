#!/bin/bash
set -o nounset

# Job settings
#BSUB -q gpuv100
#BSUB -J sm_70
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00

# Program settings
# Where to install LLVM
export LLVMDIR=/work3/s174515/LLVMV100
# Where to find the source files
export PROJECTDIR=`pwd`
# What architecture to install except for "host"
export VENDOR=NVPTX # or NVPTX
# What subarchitecture to use
export GPUARCH=sm_70

module load cmake/3.23.2
module load gcc/11.3.0-binutils-2.38
module load cuda/11.5
export PATH=/work3/s174515/ninja:$PATH

export CC=`which gcc`
export CXX=`which g++`

if [[ "$VENDOR" == "AMDGPU" ]]; then
    export OMPTARGET=amdgcn-amd-amdhsa
fi

if [[ "$VENDOR" == "NVPTX" ]]; then
    export OMPTARGET=nvptx64
fi

# Making bin
mkdir -p $PROJECTDIR/bin

## Cloning LLVM source files
if [ -d "$LLVMDIR/llvm-project" ] 
then
    echo "Found existing LLVM source code in $LLVMDIR"
else
    rm -rf $LLVMDIR
    mkdir -p $LLVMDIR
    cd $LLVMDIR
    git init
    git clone https://github.com/llvm/llvm-project
fi
cd $LLVMDIR/llvm-project
#git checkout 3d8010363895bd063a2d33172a07985b4c9b97ee
#git apply $PROJECTDIR/patches/D156263.diff

# Installing LLVM for VENDOR and GPUARCH
cd $PROJECTDIR
export LIBC_GPU_VENDOR_MATH=ON
export LIBC_GPU_BUILTIN_MATH=OFF
source install_libc.sh

# Running CPU tests
cd $PROJECTDIR 
./cpu_builtin.sh
cd $PROJECTDIR
./cpu_libc.sh

# Running GPU tests
cd $PROJECTDIR
if [[ "$VENDOR" == "AMDGPU" ]]; then
    ./gpu_ocml.sh
fi

cd $PROJECTDIR
if [[ "$VENDOR" == "NVPTX" ]]; then
    ./gpu_nv.sh
fi

# Built-in math functions
cd $PROJECTDIR
export LIBC_GPU_VENDOR_MATH=OFF
export LIBC_GPU_BUILTIN_MATH=ON
source install_libc.sh
cd $PROJECTDIR
./gpu_builtin.sh

# Built-in math functions
cd $PROJECTDIR
export LIBC_GPU_VENDOR_MATH=OFF
export LIBC_GPU_BUILTIN_MATH=OFF
source install_libc.sh
cd $PROJECTDIR
./gpu_libc.sh

