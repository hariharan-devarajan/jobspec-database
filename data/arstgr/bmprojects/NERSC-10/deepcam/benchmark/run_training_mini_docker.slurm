#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH -p ndmv4
#SBATCH --nodes=1
#SBATCH -o docker-deepcam-mini.%j
#SBATCH --job-name=docker-deepcam-mini

export UCX_IB_PCI_RELAXED_ORDERING=on \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        NCCL_DEBUG=INFO \
        NCCL_IB_PCI_RELAXED_ORDERING=1 \
        NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml \
        NCCL_SOCKET_IFNAME=eth0 \
        UCX_NET_DEVICES=eth0 \
        OMPI_MCA_coll_hcoll_enable=0

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export N10_DEEPCAM=/mnt/resource_nvme/deepcam
source $N10_DEEPCAM/deepcam_env.sh


CONT="./deepcam.sqsh"
PIN_MASK='ffffff000000,ffffff000000,ffffff,ffffff,ffffff000000000000000000,ffffff000000000000000000,ffffff000000000000,ffffff000000000000'
MOUNT="/opt/microsoft:/opt/microsoft,${N10_DEEPCAM}:/workspace,${N10_DEEPCAM}/benchmark:/benchmark,${N10_DEEPCAM}/data:/data,/opt/nccl-tests:/opt/nccl-tests"

cd $SLURM_SUBMIT_DIR
srun --mpi=pmix --cpu-bind=mask_cpu:$PIN_MASK \
	        --ntasks=$SLURM_JOB_NUM_NODES \
		--ntasks-per-node=8           \
		--gpus-per-node=8             \
                --container-mounts="${MOUNT}" \
		--container-image "${CONT}"   \
		/opt/nccl-tests/build/all_reduce_perf -b 8 -f 2 -g 1 -e 8G -c 1

sleep 2
echo "starting ..."

srun --mpi=pmix --cpu-bind=mask_cpu:$PIN_MASK \
	        --container-image "${CONT}" \
                --container-mounts="${MOUNT}" \
		--ntasks=$SLURM_JOB_NUM_NODES \
		--ntasks-per-node=8           \
		--gpus-per-node=8             \
		bash /benchmark/run_shifter.sh
