#!/bin/bash
#PBS -N caffeSetup
#PBS -l select=1:ncpus=6:mem=10gb:chip_type=e5-2680v3,walltime=6:00:00

dinstall gcc
dinstall cmake
dinstall python27
dinstall pip
dinstall numpy
dinstall ipdb

export LDFLAGS="-shared"
python -m pip install -U scipy
export LDFLAGS=""

python -m pip install -U scikit-image
dinstall opencv
dinstall leveldb
dinstall gflags
dinstall protobuf
dinstall glog
dinstall lmdb
dinstall snappy
dinstall boost
dinstall cudnn
dinstall caffe_cudnn
