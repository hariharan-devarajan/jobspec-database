#!/bin/bash
#SBATCH -A phy122
#SBATCH -t 2:00:00
#SBATCH --mail-type=ALL
## Specify these on command line sbatch -N ... -J ...
#S BATCH -N 24
#S BATCH -J xgc24

export XGC_NMPI_PER_NODE=8
export TOTAL_NMPI=$(( ${SLURM_JOB_NUM_NODES} * ${XGC_NMPI_PER_NODE} ))

NPLANES_24=4
NPLANES_128=4
NPLANES_256=8
NPLANES_512=16
NPLANES_1024=32
NPLANES_2048=64
NPLANES_4096=128
varname="NPLANES_${SLURM_JOB_NUM_NODES}"
NPLANES=${!varname}
if [ -z "$NPLANES" ]; then
    NPLANES=4
fi

echo "Nodes = ${SLURM_JOB_NUM_NODES}"
echo "Ranks = ${TOTAL_NMPI}"
echo "Planes = ${NPLANES}"

#EXE=/ccs/home/esuchyta/software/install/frontier/xgc-amd5.2-master/bin/xgc-eem-cpp-gpu
EXE=/ccs/proj/e2e/pnorbert/XGC-Devel/build.frontier/bin/xgc-eem-cpp-gpu
XGC_ROOT=/lustre/orion/phy122/world-shared/pnorbert/xgc
XGC_INPUT=${XGC_ROOT}/XGC-input

source ${XGC_ROOT}/setup-xgc.sh

function run_case () {
    MODE=${1:-bp5}
    RUN=${2:-1}
    INPUTMODE=${3:-first}
    ENGINEXML=${XGC_INPUT}/engine_${MODE}.xml
    CONF=${XGC_INPUT}/adios2cfg_in.xml;
    INPUTFILE=${XGC_INPUT}/input.${NPLANES}.${INPUTMODE}
    LOG=log-${MODE}.${RUN}

    echo "========== MODE $MODE  RUN $RUN  INPUTMODE $MODE ========="
    mkdir -p ${MODE}.${RUN}
    pushd ${MODE}.${RUN}

    if [ ! -f "${ENGINEXML}" ]; then
        echo "WARNING: xgc engine xml file ${ENGINEXML} does not exist"
    fi

    if [ ! -f "$CONF" ]; then
        echo "WARNING: adios settings file $CONF does not exist"
    fi

    if [ ! -f "${INPUTFILE}" ]; then
        echo "WARNING: XGC input file ${INPUTFILE} does not exist"
    fi

    ${XGC_INPUT}/generate_xml.sh ${ENGINEXML} ${CONF} adios2cfg.xml
    ln -s ${XGC_INPUT}
    cp ${INPUTFILE} input
    cp ${XGC_INPUT}/petsc.rc petsc.rc

    date
    echo 'srun -n ${TOTAL_NMPI} -N ${SLURM_JOB_NUM_NODES} --cpus-per-task=7 --gpus-per-node=8 --gpu-bind=closest --export=ALL,ROCM_PATH=/opt/rocm-5.2.0,OLCF_ROCM_ROOT=/opt/rocm-5.2.0,CRAYPE_LINK_TYPE=dynamic,OMP_NUM_THREADS=7,OMP_PROC_BIND=true,MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0,FI_MR_CACHE_MONITOR=memhooks,FI_CXI_RX_MATCH_MODE=software $EXE > xgc.${INPUTMODE}.log 2>&1'
    srun -n ${TOTAL_NMPI} -N ${SLURM_JOB_NUM_NODES} --cpus-per-task=7 --gpus-per-node=8 --gpu-bind=closest --export=ALL,ROCM_PATH=/opt/rocm-5.2.0,OLCF_ROCM_ROOT=/opt/rocm-5.2.0,CRAYPE_LINK_TYPE=dynamic,OMP_NUM_THREADS=7,OMP_PROC_BIND=true,MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0,FI_MR_CACHE_MONITOR=memhooks,FI_CXI_RX_MATCH_MODE=software $EXE > xgc.${INPUTMODE}.log 2>&1  
    date

    SAVEDIR=save-${INPUTMODE}
    mkdir -p ${SAVEDIR}
    cp timing* xgc.${INPUTMODE}.log ${SAVEDIR}
    cp restart_dir/xgc.restart.00001.bp/profiling.json ${SAVEDIR}/xgc.restart.00001.bp_profiling.json
    cp restart_dir/xgc.restart.f0.00001.bp/profiling.json ${SAVEDIR}/xgc.restart.f0.00001.bp_profiling.json
    cp restart_dir/xgc.restart.mvr.00001.bp/profiling.json ${SAVEDIR}/xgc.restart.mvr.00001.bp_profiling.json
    sleep 60

    popd
}

run_case null 1 first
run_case shm 1 first
run_case ews 1 first
run_case ew 1 first
run_case ew-ar1 1 first
run_case bp4 1 first
#run_case bp4 1 restart

