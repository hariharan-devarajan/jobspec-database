#!/bin/bash
#BSUB -cwd /usr/workspace/iopp/software/iopp/apps/lbann-jag
#BSUB -o /usr/workspace/iopp/software/iopp/apps/lbann-jag/out.log
#BSUB -e /usr/workspace/iopp/software/iopp/apps/lbann-jag/err.log
#BSUB -nnodes 32
#BSUB -W 6:00
#BSUB -q pbatch
#BSUB -G asccasc
#BSUB -J jag_wae

source /usr/workspace/iopp/install_scripts/bin/iopp-init
source /usr/workspace/iopp/install_scripts/bin/spack-init

spack env activate -p lbann-jag-power9le
module load spectrum-mpi/2020.08.19


echo "Started at $(date)"
recorder_lib_path=/usr/workspace/iopp/software/Recorder/install/lib/librecorder.so

jsrun --env LD_PRELOAD=$recorder_lib_path --chdir /usr/workspace/iopp/software/iopp/apps/lbann-jag --nrs 32 --rs_per_host 1 --tasks_per_rs 4 --launch_distribution packed --cpu_per_rs ALL_CPUS --gpu_per_rs ALL_GPUS lbann --prototext=/usr/workspace/iopp/software/iopp/apps/lbann-jag/experiment.prototext
status=$?
echo "Finished at $(date)"
exit ${status}
