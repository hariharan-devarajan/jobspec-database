#!/bin/bash

# Automate building PETSc and PISM on chinook.
#
# To use this script,
#
# - add "source /path/to/chinook/profile" to your .bash_profile
# - add "source /path/to/common_settings" to your .bash_profile
# - run this script

# No. of cores for make
N=8

echo 'PETSC_DIR = ' ${PETSC_DIR}
echo 'PETSC_ARCH = ' ${PETSC_ARCH}

# stop on error
set -e
# print commands before executing them
set -x

MPI_INCLUDE="/opt/scyld/openmpi/1.10.2/intel/include"
MPI_LIBRARY="/opt/scyld/openmpi/1.10.2/intel/lib/libmpi.so"

build_nco() {
    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources
    rm -rf nco
    git clone https://github.com/nco/nco.git
    cd nco
    git checkout 4.6.0

    CC=mpicc CFLAGS=-g CPPFLAGS=-I$LOCAL_LIB_DIR/include LIBS="-liomp5 -lpthread" LDFLAGS="-L$LOCAL_LIB_DIR/lib -L/usr/lib64" ./configure \
	--prefix=$LOCAL_LIB_DIR \
	--enable-netcdf-4 \
	--enable-udunits2 \
	--enable-openmp 2>&1 | tee nco_configure.log

    make -j $N  2>&1 | tee nco_compile.log
    make install  2>&1 | tee nco_install.log

}

build_petsc() {
    rm -rf $PETSC_DIR
    mkdir -p $PETSC_DIR
    cd $PETSC_DIR

    git clone --depth=1 -b maint https://gitlab.com/petsc/petsc.git .
    # Note: we use Intel compilers, disable Fortran, use 64-bit
    # indices, shared libraries, and no debugging.
    ./config/configure.py \
        --with-cc=icc --with-cxx=icpc --with-fc=0 \
        --with-blas-lapack-dir="/usr/lib64/atlas/" \
        --with-mpi-lib=$MPI_LIBRARY \
        --with-mpi-include=$MPI_INCLUDE \
        --with-64-bit-indices=1 \
        --known-mpi-shared-libraries=1 \
        --with-debugging=0 \
        --with-valgrind=0 \
        --with-x=0 \
        --with-ssl=0 \
        --with-batch=1  \
        --with-shared-libraries=1

    cat > script.queue << EOF
#!/bin/sh
 
#SBATCH --partition=t1small
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-user=aaschwanden@alaska.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=pism.%j

export PATH="$PATH:."

. /usr/share/Modules/init/sh
module load PrgEnv-intel
module load cmake/2.8.12.2
module load slurm

mpirun -np 1 ./conftest-$PETSC_ARCH
EOF

    # run conftest in an interactive job and wait for it to complete
    sbatch script.queue
    read -p "Wait for the job to complete and press RETURN."

    ./reconfigure-$PETSC_ARCH.py

    make all
}

build_petsc4py() {
    rm -rf $HOME/petsc4py
    mkdir -p $HOME/petsc4py
    cd $HOME/petsc4py
    git clone https://bitbucket.org/petsc/petsc4py.git .
    cd $HOME/petsc4py
    python setup.py build
    python setup.py install
}

build_pism() {
    mkdir -p $PISM_DIR/sources
    cd $PISM_DIR/sources

    git clone --depth 1 -b dev https://github.com/pism/pism.git . || git pull

    mkdir -p build
    cd build
    rm -f CMakeCache.txt

    # use Intel's C and C++ compilers
    export CC=icc
    export CXX=icpc
    cmake -DCMAKE_CXX_FLAGS="-O3 -ipo -axCORE-AVX2 -xSSE4.2 -diag-disable=cpu-dispatch,10006,2102" \
          -DCMAKE_C_FLAGS="-O3 -ipo -axCORE-AVX2 -xSSE4.2 -diag-disable=cpu-dispatch,10006" \
          -DCMAKE_INSTALL_PREFIX=$PISM_DIR \
          -DCMAKE_PREFIX_PATH=$LOCAL_LIB_DIR \
          -DCMAKE_PREFIX_PATH="$LOCAL_LIB_DIR;$HDF5PARALLEL_ROOT;$NETCDF_ROOT" \
          -DPism_USE_PARALLEL_NETCDF4=YES \
	  -DPism_USE_PNETCDF=YES \
          -DPism_USE_PROJ4=YES $PISM_DIR/sources
    make -j $N install
}

T="$(date +%s)"


build_nco
build_petsc
build_petsc4py
build_pism


T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"

printf "Pretty format: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"
