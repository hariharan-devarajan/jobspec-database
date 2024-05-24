#!/usr/bin/env bash

#SBATCH -A phy122
#SBATCH -o %x-%j.out
#SBATCH -t 01:30:00
#SBATCH -p batch
#SBATCH --mail-type=ALL
# Currently not configured on Frontier:
#S BATCH --ntasks-per-node=8
#S BATCH --cpus-per-task=8
#S BATCH --gpus-per-task=1
#S BATCH --gpu-bind=closest
#S BATCH -N 16
#S BATCH -J warpx

# load cray libs and ROCm libs
#export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

# From the documentation:
# Each Frontier compute node consists of [1x] 64-core AMD EPYC 7A53
# "Optimized 3rd Gen EPYC" CPU (with 2 hardware threads per physical core) with
# access to 512 GB of DDR4 memory.
# Each node also contains [4x] AMD MI250X, each with 2 Graphics Compute Dies
# (GCDs) for a total of 8 GCDs per node. The programmer can think of the 8 GCDs
# as 8 separate GPUs, each having 64 GB of high-bandwidth memory (HBM2E).

# note (5-16-22 and 7-12-22)
# this environment setting is currently needed on Frontier to work-around a
# known issue with Libfabric (both in the May and June PE)
#export FI_MR_CACHE_MAX_COUNT=0  # libfabric disable caching
# or, less invasive:
export FI_MR_CACHE_MONITOR=memhooks  # alternative cache monitor

# note (9-2-22, OLCFDEV-1079)
# this environment setting is needed to avoid that rocFFT writes a cache in
# the home directory, which does not scale.
export ROCFFT_RTC_CACHE_PATH=/dev/null

export OMP_NUM_THREADS=1
export WARPX_NMPI_PER_NODE=8
export TOTAL_NMPI=$(( ${SLURM_JOB_NUM_NODES} * ${WARPX_NMPI_PER_NODE} ))

EXE=/lustre/orion/phy122/world-shared/pnorbert/warpx/warpx
WARPX_ROOT=/ccs/proj/e2e/pnorbert/warpx
INPUT=/lustre/orion/phy122/world-shared/pnorbert/warpx/input.n${SLURM_JOB_NUM_NODES}

source $WARPX_ROOT/frontier_warpx_profile.sh

function run_case () {
    MODE=${1:-bp5}
    RUN=${2:-1}
    CONF=$WARPX_ROOT/job.frontier/parameters-${MODE}.txt
    LOG=log-${MODE}.${RUN}

    echo "========== MODE $MODE  RUN $RUN ========="
    mkdir ${MODE}.${RUN}
    pushd ${MODE}.${RUN}
    cat $INPUT $CONF >> input

    if [ ! -f "$INPUT" ]; then
        echo "WARNING: warpx input file $INPUT does not exist"
    fi

    if [ ! -f "$CONF" ]; then
        echo "WARNING: adios settings file $CONF does not exist"
    fi

    date
    echo `srun -N${SLURM_JOB_NUM_NODES} -n${TOTAL_NMPI} --ntasks-per-node=${WARPX_NMPI_PER_NODE} ${EXE} input > output.txt`
    srun -N${SLURM_JOB_NUM_NODES} -n${TOTAL_NMPI} --ntasks-per-node=${WARPX_NMPI_PER_NODE} ${EXE} input > output.txt
    date
    sleep 30

    popd
}

run_case nullcore 1
run_case ews 1
run_case ew 1
run_case shm 1
run_case ew-ar1 1
