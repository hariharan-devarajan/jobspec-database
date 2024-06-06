#!/bin/bash
# Begin LSF Directives
#BSUB -P FUS123
#BSUB -W 2:00
#BSUB -nnodes 5
#BSUB -J XGC
#BSUB -o XGC.%J
#BSUB -e XGC.%J

rmAdiosFiles() {
  rm -rf *.bp
  rm -rf *.sst
}

run_coupler() {
  local outfile=$1
	module load nvhpc/21.7
	module load spectrum-mpi/10.4.0.3-20210112
	module load netlib-lapack/3.9.1
	module load hypre/2.22.0-cpu
	module load fftw/3.3.9
	module load hdf5/1.10.7
	module load cmake/3.20.2
	module load libfabric/1.12.1-sysrdma


  #system adios2 install {
  #from https://github.com/ornladios/ADIOS2/issues/2887#issuecomment-1021428076
  module load adios2/2.7.1
  local rdmaVars="-EFABRIC_IFACE=mlx5_0 \
                  -EOMPI_MCA_coll_ibm_skip_barrier=true \
                  -EFI_MR_CACHE_MAX_COUNT=0 \
                  -EFI_OFI_RXM_USE_SRX=1"
  #}

  set -x
	jsrun -n 2 -r 2 -a 1 -g 1 -c 1 -b rs -EOMP_NUM_THREADS=1 ${rdmaVars} /gpfs/alpine/scratch/jmerson/fus123/buildWDMApp/test/xgc_n0_server 123mesh.osh 123mesh_2.cpn 8 >>${outfile} &
  set +x
  module purge
}

run_xgc() {
  local outfile=$1
	module load nvhpc/21.7
	module load spectrum-mpi/10.4.0.3-20210112
	module load netlib-lapack/3.9.1
	module load hypre/2.22.0-cpu
	module load fftw/3.3.9
	module load hdf5/1.10.7
	module load cmake/3.20.2
	module load libfabric/1.12.1-sysrdma


  rmAdiosFiles

  #system adios2 install {
  #from https://github.com/ornladios/ADIOS2/issues/2887#issuecomment-1021428076
  module load adios2/2.7.1
  local rdmaVars="-EFABRIC_IFACE=mlx5_0 \
                  -EOMPI_MCA_coll_ibm_skip_barrier=true \
                  -EFI_MR_CACHE_MAX_COUNT=0 \
                  -EFI_OFI_RXM_USE_SRX=1"
  #}

  set -x
	#jsrun -n 8 -r 4 -a 1 -g 1 -c 1 -b rs -EOMP_NUM_THREADS=1 ${rdmaVars} /gpfs/alpine/scratch/jmerson/fus123/xgc_delta_f/bin/xgc-es-cpp-gpu &>>${outfile} &
	jsrun -n 8 -r 8 -a 1 -g 0 -c 1 -b rs -EOMP_NUM_THREADS=1 ${rdmaVars} /gpfs/alpine/scratch/jmerson/fus123/xgc_delta_f/bin/xgc-es-cpp &>>${outfile} &
  set +x
  module purge
}

echo $(date)

root=$PWD

echo "LSB_MCPU_HOSTS ${LSB_MCPU_HOSTS}"
ROOT_DIR=$PWD

cd $ROOT_DIR/deltaf1
run_xgc deltaf1.${LSB_JOBID}.out
cd $ROOT_DIR/deltaf2
run_xgc deltaf2.${LSB_JOBID}.out
cd $ROOT_DIR
run_coupler coupler.${LSB_JOBID}.out


jswaitall
wait
