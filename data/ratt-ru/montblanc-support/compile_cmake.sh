#PBS -q GPUQ
#PBS -l nodes=srvslsgpu003:ppn=4:seriesGPUk
#PBS -N compile-cmake
#PBS -o Q.compile-cmake.log
#PBS -e Q.compile-cmake.err
#PBS -V

source $HOME/scripts/base.sh

module load compilers/gcc492
module load mpi/openmpi-1.8.8
module load cuda/Cuda-6.5

cd ${SOURCE_PREFIX}/cmake-3.3.2
./configure --prefix=${INSTALL_PREFIX}
make clean
make
make install
