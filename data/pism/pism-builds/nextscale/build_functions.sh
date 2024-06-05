#!/bin/bash

# Automate building PETSc and PISM on PIK IBM NeXtScale.
# Following the build script by Constantine Khroulev and Andy Aschwanden (UAF)
#
# To use this script,
#
# - add "source /path/to/nextscale/profile" to your .bash_profile
# - run this script

# No. of cores for make
N=2

echo 'PETSC_DIR = ' ${PETSC_DIR}
echo 'PETSC_ARCH = ' ${PETSC_ARCH}

MPI_INCLUDE="${I_MPI_ROOT}/include64"
MPI_LIBRARY="${I_MPI_ROOT}/lib64/libmpi.so"
MPI_DIR="${I_MPI_ROOT}/intel64"

if [ $onbroadwell = True ]; then
   PART="srun --partition=broadwell --exclusive --ntasks 1"
else
   PART=""
fi
#echo $PART

build_hdf5() {

    # download and build HDF5
    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    name=hdf5-1.10.2
    url=https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/${name}/src/${name}.tar.gz


    wget -nc ${url}
    tar xzvf ${name}.tar.gz

    cd ${name}
    CC=$CC ./configure --enable-parallel \
           --with-zlib=/p/system/slurmdeps/17.11.2/build/zlib-1.2.11 \
           --prefix=$LOCAL_LIB_DIR/hdf5${addname} 2>&1 | tee hdf5_configure.log

    make all -j $N 2>&1 | tee hdf5_compile.log
    make install 2>&1 | tee hdf5_install.log
}

build_netcdf() {

    # download and build netcdf

    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources
    version=4.6.1
    url=ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-${version}.tar.gz

    wget -nc ${url}
    tar -zxvf netcdf-${version}.tar.gz

    cd netcdf-${version}
    #export CC=mpiicc
    export CPPFLAGS="-I$LOCAL_LIB_DIR/hdf5${addname}/include"
    export LDFLAGS=-L$LOCAL_LIB_DIR/hdf5${addname}/lib
    $PART ./configure \
      --enable-netcdf4 \
      --disable-dap \
      --prefix=$LOCAL_LIB_DIR/netcdf${addname} 2>&1 | tee netcdf_configure.log
    # --with-zlib=/p/system/slurmdeps/17.11.2/build/zlib-1.2.11 \
    $PART make all -j $N 2>&1 | tee netcdf_compile.log
    $PART make install 2>&1 | tee netcdf_install.log
}


build_proj4 () {

    # download and build PROJ.4
    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    version=5.0.1
    url=http://download.osgeo.org/proj/proj-${version}.tar.gz    

    wget -nc ${url}
    tar -zxvf proj-${version}.tar.gz
    cd proj-${version}

    # remove and re-generate files created by autoconf
    autoreconf --force --install
    ./configure --prefix=$LOCAL_LIB_DIR/proj-${version}

    make all
    make install
}

build_udunits() {

    # download and build UDUNITS
    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    version=2.2.26
    url=ftp://ftp.unidata.ucar.edu/pub/udunits/udunits-${version}.tar.gz

    wget -nc ${url}
    tar xzvf udunits-${version}.tar.gz

    cd udunits-${version}

    $PART ./configure \
      --prefix=$LOCAL_LIB_DIR/udunits${addname} 2>&1 | tee udunits_configure.log

    $PART make all -j $N 2>&1 | tee udunits_compile.log
    $PART make install 2>&1 | tee udunits_install.log

}

build_gsl() {

    # download and build GSL
    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    version=2.4
    url=ftp://ftp.gnu.org/gnu/gsl/gsl-${version}.tar.gz

    wget -nc ${url}
    tar xzvf gsl-${version}.tar.gz

    cd gsl-${version}

    $PART ./configure \
      --prefix=$LOCAL_LIB_DIR/gsl${addname} 2>&1 | tee gsl_configure.log

    $PART make all -j $N 2>&1 | tee gsl_compile.log
    $PART make install 2>&1 | tee gsl_install.log

}

build_fftw() {

    # download and build FFTW
    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    version=3.3.7
    url=ftp://ftp.fftw.org/pub/fftw/fftw-${version}.tar.gz

    wget -nc ${url}
    tar xzvf fftw-${version}.tar.gz

    cd fftw-${version}

    $PART ./configure \
      --prefix=$LOCAL_LIB_DIR/fftw${addname} \
      --with-pic \
      --enable-mpi \
      CC=$CC MPICC=$CC 2>&1 | tee fftw_configure.log

    $PART make -j $N 2>&1 | tee fftw_compile.log
    $PART make install 2>&1 | tee fftw_install.log

}


build_petsc() {

    rm -rf $PETSC_DIR
    mkdir -p $PETSC_DIR
    cd $PETSC_DIR

    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    echo $PETSC_DIR

    version=3.9.1
    #version=3.8.3
    url=http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-${version}.tar.gz

    wget -nc ${url}
    tar xzvf petsc-${version}.tar.gz

    cp -r petsc-${version}/* $PETSC_DIR
    cd $PETSC_DIR

    echo $MKLROOT
    echo $CFLAGS
    

    # Note: we use Intel compilers, disable Fortran, use 64-bit
    # indices, shared libraries, and debugging.
    $PART ./config/configure.py \
        --with-prefix=$PETSC_DIR \
        --with-cc=mpiicc --with-cxx=mpiicpc --with-fc=0 \
        --with-mpi-lib=$MPI_LIBRARY \
        --with-mpi-include=$MPI_INCLUDE \
        --with-blas-lapack-dir="$MKLROOT/lib/intel64" \
        --with-64-bit-indices=1 \
        --known-64-bit-blas-indices \
        --known-mpi-shared-libraries=1 \
        --with-debugging=1 \
        --with-valgrind=1 \
        --with-valgrind-dir=$VGROOT \
        --with-x=0 \
        --with-ssl=0 \
        --with-batch=1  \
        --with-shared-libraries=1 \
        --with-mpiexec="srun" \
          CFLAGS="$CFLAGS" \
          CXXFLAGS="$CXXFLAGS" 

          #--prefix=$PETSC_INSTALL_DIR

    #    COPTFLAGS="-g -O3 -ffast-math -march=bdver1 -fPIC" \
    #    FOPTFLAGS="-g -O3 -ffast-math -march=bdver1 -fPIC"


    # run conftest in an interactive job and wait for it to complete
    #srun -t 20 -n 1 -N 1 ./conftest-$PETSC_ARCH

    # run conftest in aa subitted job and wait for it to complete
    cat > conftest-slurm.sh <<EOF
#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=petsc_config
#SBATCH --account=ice
#SBATCH --output=./petsc_config-%j.out
#SBATCH --error=./petsc_config-%j.err
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=albrecht@pik-potsdam.de

cd $PETSC_DIR
#srun -n 1 ./conftest-$PETSC_ARCH
if [ $onbroadwell = True ]; then
   $PART ./conftest-$PETSC_ARCH
else
   srun -n 1 ./conftest-$PETSC_ARCH
fi
EOF

    sbatch ./conftest-slurm.sh

    #read -p "Wait for the job to complete and press RETURN."
    #read -p "Continuing in 10 Seconds...." -t 10
    echo "Continuing in 30 Seconds...."
    sleep 30
    echo "Continuing ...."

    $PART ./reconfigure-$PETSC_ARCH.py

    $PART make all 
    $PART make test
}


build_swig() {

    # download and build SWIG
    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    version=3.0.12
    url=https://sourceforge.net/projects/swig/files/swig/swig-${version}/swig-${version}.tar.gz

    wget ${url}
    tar xzvf swig-${version}.tar.gz
    cd swig-${version}

    $PART ./configure \
      --prefix=$LOCAL_LIB_DIR/swig${addname} 2>&1 | tee swig_configure.log

    $PART make -j $N 2>&1 | tee swig_compile.log
    $PART make install 2>&1 | tee swig_install.log

}


build_petsc4py() {

    module load anaconda/5.0.0
    # create a new virtual environment, with numpy already installed
    #conda create --name python_for_pism_calib numpy
    source activate python_for_pism_calib
    #pip install cython
    #which python

    #unset CFLAGS
    #unset CXXFLAGS

    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    version=3.9.1
    url=https://bitbucket.org/petsc/petsc4py/petsc4py-${version}.tar.gz

    wget ${url}
    tar xzvf petsc4py-${version}.tar.gz

    PETSC4PYPATH=$LOCAL_LIB_DIR/petsc4py${addname}
    rm -rf $PETSC4PYPATH
    #mkdir -p $PETSC4PYPATH
    cp -r petsc4py-${version} $PETSC4PYPATH
    cd $PETSC4PYPATH

    #git clone https://bitbucket.org/petsc/petsc4py.git $PETSC4PYPATH
    #cd $PETSC4PYPATH
    python setup.py install --user
    #python setup.py build
    #python setup.py install
}


build_pism() {

    set -e
    set -x
    #mkdir -p $PISM_DIR/sources
    #cd $PISM_DIR/sources

    mkdir -p $LOCAL_LIB_DIR/sources
    cd $LOCAL_LIB_DIR/sources

    git clone --depth 1 -b master https://github.com/pism/pism.git || git pull

    cd pism
    rm -rf build
    mkdir -p build
    cd build

    # use Intel's C and C++ compilers
    export PATH=$LOCAL_LIB_DIR/fftw${addname}/lib/:$LOCAL_LIB_DIR/fftw${addname}/include/:$PATH
    export PATH=$LOCAL_LIB_DIR/gsl${addname}/lib/:$LOCAL_LIB_DIR/gsl${addname}/include/:$PATH
    export PATH=$LOCAL_LIB_DIR/udunits${addname}/lib/:$LOCAL_LIB_DIR/udunits${addname}/include/:$PATH
    export PATH=$LOCAL_LIB_DIR/doxygen/bin/:$PATH
    #export PATH=/p/system/packages/hdf5/1.8.18/impi/lib/:$PATH
    export PATH=$HDF5ROOT/lib/:$PATH

    #if build with petsc4py:
    #export PATH=$LOCAL_LIB_DIR/swig${addname}/:$PATH
    #export PETSC4PY_LIB=/home/albrecht/software/petsc4py-3.8.1/lib/python2.7/site-packages
    #export PYTHONPATH=$PETSC4PY_LIB:${PYTHONPATH}

    cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
          -DCMAKE_INSTALL_PREFIX=$PISM_DIR \
          -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
          -DCMAKE_C_FLAGS="$CFLAGS" \
          -DCMAKE_CXX_COMPILER=mpiicpc \
          -DCMAKE_C_COMPILER=mpiicc  \
          -DPETSC_EXECUTABLE_RUNS=ON \
          -DMPI_C_DIR="$MPI_DIR" \
          -DPism_USE_JANSSON=NO \
          -DPism_USE_PARALLEL_NETCDF4=YES \
          -DPism_USE_PROJ4=YES \
          -DPism_BUILD_EXTRA_EXECS:BOOL=OFF \
          -DMPIEXEC=/p/system/slurm/bin/srun \
           $LOCAL_LIB_DIR/sources/pism \
          --debug-trycompile ../.
    make -j $N install
    set +x
    set +e

    #if build with petsc4py:

          #-DPETSC_EXECUTABLE_RUNS=ON \
          #-DPism_BUILD_EXTRA_EXECS:BOOL=ON \
          #-DMPI_C_INCLUDE_PATH=${I_MPI_ROOT}/include64 \
          #-DMPI_C_LIBRARIES=${I_MPI_ROOT}/lib64/libmpi.so \
          #-DHDF5_INCLUDE_DIRS:PATH=$HDF5ROOT/include \
          #-DHDF5_LIBRARIES:PATH=$HDF5ROOT/lib/libhdf5.so \
          #-DHDF5_hdf5_LIBRARY_RELEASE:FILEPATH=$HDF5ROOT/lib/libhdf5.so \
          #-DHDF5_hdf5_hl_LIBRARY_RELEASE:FILEPATH=$HDF5ROOT/lib/libhdf5_hl.so \
          #-DHDF5_z_LIBRARY_RELEASE:FILEPATH=/usr/lib64/libz.so \

    #export PYTHONPATH=/home/albrecht/.conda/envs/python_for_pism_calib/lib/python2.7/site-packages::$PYTHONPATH
    #export PYTHONPATH=$PISM_INSTALL_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    #export PYTHONPATH=$PISM_INSTALL_PREFIX/bin:${PYTHONPATH}

}

build_all() {

    #module load pism/stable1.0

    build_hdf5      # or: module load hdf5/1.10.2/intel/parallel
    build_netcdf    # or: module load netcdf-c/4.6.1/intel/parallel

    build_udunits   # or: module load udunits/2.2.26
    build_gsl       # or: module load gsl/2.4
    build_fftw      # or: module load fftw/3.3.7
    build_proj4     # or: module load proj4/5.0.1

    build_petsc      # or: PETSC_DIR

    build_swig
    #build_petsc4py

    #build_pism
}
