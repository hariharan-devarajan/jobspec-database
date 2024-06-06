#!/bin/bash
#BSUB -cwd /usr/workspace/iopp/software/iopp/apps/lbann-atom/1-node
#BSUB -o /usr/workspace/iopp/software/iopp/apps/lbann-atom/1-node/out.log
#BSUB -e /usr/workspace/iopp/software/iopp/apps/lbann-atom/1-node/err.log
#BSUB -nnodes 1
#BSUB -W 1:00
#BSUB -J atom
#BSUB -q pdebug
#BSUB -G asccasc

source /usr/workspace/iopp/install_scripts/bin/iopp-init
source /usr/workspace/iopp/install_scripts/bin/spack-init

spack env activate -p lbann-atom-power9le
module load spectrum-mpi/2020.08.19

export LBANN_USE_CUBLAS_TENSOR_OPS=1
export LBANN_USE_CUDNN_TENSOR_OPS=1
export MV2_USE_RDMA_CM=0
export AL_PROGRESS_RANKS_PER_NUMA_NODE=2
export OMP_NUM_THREADS=8
export IBV_FORK_SAFE=1
export HCOLL_ENABLE_SHARP=0
export OMPI_MCA_coll_hcoll_enable=0
export PAMI_MAX_NUM_CACHED_PAGES=0
export NVSHMEM_MPI_LIB_NAME=libmpi_ibm.so
echo "Started at $(date)"
#recorder_lib_path=/usr/workspace/iopp/software/Recorder/install/lib/librecorder.so
export RECORDER_EXCLUSION_FILE=$HOME/exclusion_prefix.txt

jsrun --env LD_PRELOAD=$recorder_lib_path --bind packed:8 --smpiargs="-gpu"     \
        --chdir /usr/workspace/iopp/software/iopp/apps/lbann-atom/1-node        \
        --nrs 1 --rs_per_host 1 --tasks_per_rs 4 --launch_distribution packed   \
        --cpu_per_rs ALL_CPUS --gpu_per_rs ALL_GPUS                             \
        /usr/WS2/iopp/software/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/lbann-atom-power9le-temkp6m4m55t3kiz2cfutcsegbqsu3zu/bin/lbann \
        --vocab=/p/gpfs1/brainusr/datasets/atom/combo_enamine1613M_mpro_inhib/enamine_all2018q1_2020q1-2_mpro_inhib_kekulized.vocab          \
        --data_filedir=/p/gpfs1/brainusr/datasets/atom/combo_enamine1613M_mpro_inhib/                                                        \
        --data_filename_train=test_enamine_all2018q1_2020q1-2_mpro_inhib_kekulized_2010kSMILES.csv                                           \
        --sequence_length=100  --num_io_threads=11 --delimiter=c --use_data_store --preload_data_store                                       \
        --procs_per_trainer=4 --prototext=/usr/workspace/iopp/software/iopp/apps/lbann-atom/1-node/experiment.prototext --verbose

status=$?
echo "Finished at $(date)"
exit ${status}
