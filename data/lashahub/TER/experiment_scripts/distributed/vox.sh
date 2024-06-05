#!/bin/bash
#SBATCH --comment clap
#SBATCH --partition=all
#SBATCH --job-name=mclap
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --output=result.txt

nvidia-smi

export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

# Set up environment variables for distributed training
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo go $COUNT_NODE
echo $HOSTNAMES

# Assuming you have previously activated your environment
# source /mnt/beegfs/home/koroshinadze/miniconda3/envs/clap/bin/activate

cd /mnt/beegfs/home/koroshinadze/sketching_audio/TER/CLAP/src/laion_clap
export TRANSFORMERS_CACHE=/mnt/beegfs/home/koroshinadze/transformers_cache

echo "Starting training using torchrun"
export PYTHONPATH="/mnt/beegfs/home/koroshinadze/sketching_audio/TER/CLAP/src:$PYTHONPATH"

# Using torchrun to start the training
torchrun --nproc_per_node=4 --nnodes=$COUNT_NODE --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT evaluate/eval_linear_probe.py \
    --save-frequency 50 \
    --save-top-performance 4 \
    --save-most-recent \
    --dataset-type="webdataset" \
    --precision="fp32" \
    --warmup 0 \
    --batch-size=160 \
    --lr=1e-4 \
    --wd=0.1 \
    --epochs=100 \
    --workers=3 \
    --use-bn-sync \
    --freeze-text \
    --amodel HTSAT-tiny \
    --tmodel roberta \
    --datasetnames "vocalimitations" \
    --datasetpath "/mnt/beegfs/home/koroshinadze/sketching_audio/TER/CLAP/dataset" \
    --datasetinfos "train" \
    --seed 3407 \
    --logs /mnt/beegfs/home/koroshinadze/clap_logs \
    --gather-with-grad \
    --lp-loss="ce" \
    --lp-metrics="acc" \
    --lp-lr=1e-4 \
    --lp-mlp \
    --class-label-path="/mnt/beegfs/home/koroshinadze/sketching_audio/TER/CLAP/dataset/vocalimitations/labels.json" \
    --openai-model-cache-dir /mnt/beegfs/home/koroshinadze/transformers_cache \
    --pretrained="/mnt/beegfs/home/koroshinadze/sketching_audio/TER/CLAP/pretrained" \
    --top-k-checkpoint-select-dataset="vocalimitations-test" \
    --top-k-checkpoint-select-metric="mAP@10" \
    --data-filling "repeatpad" \
    --data-truncating "rand_trunc" \
    --optimizer "adam" \
    --data-truncating "fusion" \
    --enable-fusion \
    --fusion-type "aff_2d" \
    --pretrained-audio "/mnt/beegfs/home/koroshinadze/sketching_audio/TER/CLAP/pretrained/HTSAT-fullset-imagenet-tiny-map=0.467.ckpt"
