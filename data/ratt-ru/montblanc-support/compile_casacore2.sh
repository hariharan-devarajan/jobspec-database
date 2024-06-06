#PBS -q GPUQ
#PBS -l nodes=srvslsgpu003:ppn=4:seriesGPUk
#PBS -N compile-casacore2
#PBS -o Q.compile-casacore2.log
#PBS -e Q.compile-casacore2.err
#PBS -V

source $HOME/scripts/base.sh

module load compilers/gcc492
module load mpi/openmpi-1.8.8
module load cuda/Cuda-6.5

source $HOME/scripts/compiler_vars.sh

cd $SOURCE_PREFIX/casacore-2.0.3
rm -rf build && mkdir -p build && cd build
#mkdir -p build && cd build
CMAKE_OPTIONS=""
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_CXX_COMPILER=${MPICPP}"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_C_COMPILER=${MPICC}"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCMAKE_Fortran_COMPILER=${MPIFORT}"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DUSE_FFTW3=ON"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DUSE_OPENMP=ON"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DUSE_HDF5=ON"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DBUILD_PYTHON=ON"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DUSE_THREADS=ON"
CMAKE_OPTIONS="${CMAKE_OPTIONS} -DCXX11=ON"

cmake .. ${CMAKE_OPTIONS}

make -j 4
make -j 4 install

cd ${SOURCE_PREFIX}/python-casacore-2.0.0
python setup.py build
python setup.py install --prefix=${INSTALL_PREFIX}
