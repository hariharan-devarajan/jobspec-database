#!/bin/bash
#SBATCH --job-name=deepcam-mini
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x-%j.out
#SBATCH -p ndmv4

#load the DeepCAM environment
#N10_DEEPCAM is defined in deepcam_env.sh
#module load python
export N10_DEEPCAM=/mnt/resource_nvme/deepcam
source $N10_DEEPCAM/deepcam_env.sh
module load mpi/hpcx
#load hyperparameters
#BENCH_RCP is defined by bench_rcp.conf
#this reference convergence point (RCP) should not be modified
source bench_rcp.conf

#the local batch size may be adjusted
#under the constraint that the global batch size is fixed to 2048,
#i.e. processes * local_batch_size = 2048.
#for example: local_batch_size=$(( 2048 / ${SLURM_NTASKS} ))
local_batch_size=2

#other options within this script may be adjusted freely
data_dir=$N10_DEEPCAM/data/mini
output_dir=$N10_DEEPCAM/output_dir
run_tag="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

sudo ./max_gpu_app_clocks.sh -l
sudo ./max_gpu_app_clocks.sh

mpirun -np 8 --map-by ppr:8:node:PE=12 -rank-by slot --report-bindings -mca coll ^hcoll \
    -x LD_LIBRARY_PATH \
    -x PATH            \
    -x CUDA_VISIBLE_DEVICES=2,3,0,1,6,7,4,5 \
    -x NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml \
    -x NCCL_IB_PCI_RELAXED_ORDERING=1 \
    -x UCX_IB_PCI_RELAXED_ORDERING=on \
    -x CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 $N10_DEEPCAM/baseline/src_deepCam/train.py \
    ${BENCH_RCP_BASELINE} \
    --wireup_method "nccl-openmpi" \
    --run_tag ${run_tag} \
    --data_dir_prefix ${data_dir} \
    --output_dir ${output_dir} \
    --model_prefix "segmentation" \
    --optimizer "AdamW" \
    --max_epochs 64 \
    --max_inter_threads 1 \
    --local_batch_size ${local_batch_size}

sudo ./max_gpu_app_clocks.sh -r
#save results for successful run
if [[ $? == 0 ]]; then
   mkdir -p $N10_DEEPCAM/results/jobscripts
   mkdir -p $N10_DEEPCAM/results/logs
   if [[ $SLURM_JOB_QOS != interactive ]] && [[ SLURM_JOB_NAME != interactive ]]; then
       cp ${0} $N10_DEEPCAM/results/jobscripts/${SLURM_JOB_ID}.slurm
       cp -p $output_dir/logs/${run_tag}.log $N10_DEEPCAM/results/logs 
   fi
fi

