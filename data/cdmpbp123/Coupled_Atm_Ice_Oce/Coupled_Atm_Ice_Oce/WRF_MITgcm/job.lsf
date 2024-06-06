#!/bin/bash
#BSUB -q production
#BSUB -n 32
#BSUB -R "span[ptile=28]"
#BSUB -J ctrlrun
#BSUB -o %J.out
#BSUB -e %J.err

export I_MPI_HYDRA_BOOTSTRAP=lsf
export I_MPI_HYDRA_BRANCH_COUNT=12
export I_MPI_LSF_USE_COLLECTIVE_LAUNCH=1
. /fs01/platform/lsf/conf/profile.lsf
export INTEL_LICENSE_FILE=/fs01/apps/intel/COM_L___L9TX-FXGWPC8V.lic
export LSF_SERVERDIR=/fs01/platform/lsf/9.1/linux2.6-glibc2.3-x86_64/etc
export LSF_LIBDIR=/fs01/platform/lsf/9.1/linux2.6-glibc2.3-x86_64/lib
export LSF_BINDIR=/fs01/platform/lsf/9.1/linux2.6-glibc2.3-x86_64/bin
export LSF_ENVDIR=/fs01/platform/lsf/conf
export XLSF_UIDDIR=/fs01/platform/lsf/9.1/linux2.6-glibc2.3-x86_64/lib/uid
export PATH=/apps/apps/intel/impi/2017.2.191/bin64:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/mpi/intel64/bin:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/bin/intel64:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/mpi/intel64/bin:/fs01/apps/intel/debugger_2017/gdb/intel64_mic/bin:/fs01/platform/lsf/9.1/linux2.6-glibc2.3-x86_64/etc:/fs01/platform/lsf/9.1/linux2.6-glibc2.3-x86_64/bin:/opt/ibm/toolscenter/asu:/usr/lpp/mmfs/bin:/usr/lib64/qt-3.3/bin:/usr/kerberos/sbin:/usr/kerberos/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/opt/ibutils/bin:/fs01/apps:/fs01/apps/nco-4.6.7/bin:/fs01/apps/ImageMagick-7.0.6-5/bin:/fs01/apps/ncview-2.1.7/bin:/fs01/apps/ncl-6.4.0/bin:/fs01/apps/netcdf-4.4.4-ifort/bin:/fs01/apps/netcdf-4.4.1.1-gcc/bin:/usr/local/MATLAB/R2017a/bin:/fs01/home/lomf/software/grads-2.1.0.oga.1/Contents:/fs01/home/lomf/bin:/fs01/home/lomf/person/rensh/C-Coupler2_from_yh/model_platform/scripts/utils/

export LD_LIBRARY_PATH=/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/compiler/lib/intel64_lin:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/mkl/lib/intel64_lin:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/mpi/intel64/lib:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/mpi/mic/lib:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/compiler/lib/intel64:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/compiler/lib/intel64_lin:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/mpi/intel64/lib:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/mpi/mic/lib:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/ipp/lib/intel64:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/compiler/lib/intel64_lin:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/mkl/lib/intel64_lin:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/tbb/lib/intel64/gcc4.4:/fs01/apps/intel/debugger_2017/iga/lib:/fs01/apps/intel/debugger_2017/libipt/intel64/lib:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/daal/lib/intel64_lin:/fs01/apps/intel/compilers_and_libraries_2017.3.191/linux/daal/../tbb/lib/intel64_lin/gcc4.4:/fs01/platform/lsf/9.1/linux2.6-glibc2.3-x86_64/lib:/fs01/apps/netcdf-4.4.4-ifort/lib:/fs01/apps/netcdf-4.4.1.1-icc/lib:/fs01/apps/libpng-1.6.30/lib:/lib64/:/apps/apps/intel/mkl/lib/intel64:/fs01/apps/netcdf-4.4.1.1-icc/lib
source /apps/apps/intel/mkl/bin/mklvars.sh intel64
source /fs01/apps/intel/bin/compilervars.sh intel64
source /apps/apps/intel/impi/2017.2.191/bin64/mpivars.sh intel64


mpiexec.hydra -genv I_MPI_DEVICE rdma -np 16  "/fs01/home/lomf/person/rensh/C-Coupler2_from_yh/WRF_MITgcm/run/cpl/mitgcm/exe/mitgcm"  :  -np 16  "/fs01/home/lomf/person/rensh/C-Coupler2_from_yh/WRF_MITgcm/run/atm/wrf/exe/wrf" 
