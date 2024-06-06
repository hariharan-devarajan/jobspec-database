#!/bin/sh
# vim: set tw=120:
#BSUB -P fus123
#BSUB -q debug
#BSUB -W 00:10
#BSUB -nnodes 1
#BSUB -alloc_flags "smt4"
#BSUB -J setupComms
#BSUB -o %J.out
#BSUB -e %J.out

run() {
  local nodes=$1
  local processes=$2
  local exe=$3
  local args=$4
  local outfile=$5
  local isSst=$6

  module load spectrum-mpi/10.4.0.3-20210112 gcc/10.2.0

  #system adios2 install {
  #from https://github.com/ornladios/ADIOS2/issues/2887#issuecomment-1021428076
  module load adios2/2.7.1
  local rdmaVars=""
  if [ ${isSst} == 1 ]; then
    rdmaVars="-EFABRIC_IFACE=mlx5_0 \
              -EOMPI_MCA_coll_ibm_skip_barrier=true \
              -EFI_MR_CACHE_MAX_COUNT=0 \
              -EFI_OFI_RXM_USE_SRX=1 \
              -ESstVerbose=5"
  fi
  echo "rdmaVars $rdmaVars"
  #}

  set -x
  local ranksPerNode=$((processes/nodes))
  jsrun --nrs ${processes} \
    --tasks_per_rs 1 \
    --cpu_per_rs 1 \
    --gpu_per_rs 0 \
    --rs_per_host ${ranksPerNode} \
    --latency_priority CPU-CPU \
    --launch_distribution packed \
    --bind packed:1 \
    -EOMP_NUM_THREADS=1 \
    ${rdmaVars} \
    ${exe} ${args} &>> ${outfile} &
  set +x
  module purge
}

echo $(date)

rmAdiosFiles() {
  rm -rf *.bp
  rm -rf *.sst
}

rmAdiosFiles

echo "LSB_MCPU_HOSTS ${LSB_MCPU_HOSTS}"
numJobNodes=$(echo ${LSB_MCPU_HOSTS} | grep -o -i ' [a-z][0-9]\+n'  | wc -l)

bin=/gpfs/alpine/fus123/scratch/cwsmith/twoClientWdmCplTesting/buildAdios2SstTest_sysAdios2/setupComms

serverRanks=1
serverNodes=1

clientRanks=1
clientNodes=1

isSst=1

clientExe=${bin}
clientArgs="${isSst} 0"
clientOut=clientSst_${LSB_JOBID}.out
run $clientNodes $clientRanks $clientExe "$clientArgs" $clientOut ${isSst}

serverExe=${bin}
serverArgs="${isSst} 1"
serverOut=serverSst_${LSB_JOBID}.out
run $serverNodes $serverRanks $serverExe "$serverArgs" $serverOut ${isSst}

set -x    
jswait all
rmAdiosFiles
set +x

isSst=0

clientExe=${bin}
clientArgs="${isSst} 0"
clientOut=clientBp4_${LSB_JOBID}.out
run $clientNodes $clientRanks $clientExe "$clientArgs" $clientOut ${isSst}

serverExe=${bin}
serverArgs="${isSst} 1"
serverOut=serverBp4_${LSB_JOBID}.out
run $serverNodes $serverRanks $serverExe "$serverArgs" $serverOut ${isSst}

set -x    
jswait all
rmAdiosFiles
set +x

wait

