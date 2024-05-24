#!/bin/bash

#SBATCH --job-name=train_cac
#SBATCH --partition=gpu_high
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --chdir=/work/rspandey/train_iais/

NODE_LIST=$( scontrol show hostname $SLURM_JOB_NODELIST | sed -z 's/\n/\:4,/g' )
NODE_LIST=${NODE_LIST%?}
echo $NODE_LIST

DOWNLOADS="/work/rspandey/train_iais/downloads" ;
singularity exec --nv \
    -B $(pwd):/src \
    -B $DOWNLOADS/finetune:/storage \
    -B $DOWNLOADS/pretrained:/pretrain \
    -B $DOWNLOADS/txt_db:/txt \
    -B $DOWNLOADS/img_db:/img \
    --env NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --pwd /src /work/rspandey/uniter.sif \
    mpirun -np $SLURM_NTASKS -H $NODE_LIST -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^lo -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_openib_verbose 1 python train_itm_hard_negatives.py --config config/train-siais-large.json --IAIS soft --num_train_steps 5000 --valid_steps 1000 --tsa_schedule exp_schedule
    # mpirun -np $SLURM_NTASKS -H $NODE_LIST -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^lo -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib -mca btl_openib_verbose 1 -mca btl_tcp_if_incle 192.168.0.0/16 -mca oob_tcp_if_include 192.168.0.0/16 python train_itm_hard_negatives.py --config config/train-siais-large.json --IAIS soft --num_train_steps 5000 --valid_steps 1000 --tsa_schedule exp_schedule

# ### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
# ### change WORLD_SIZE as gpus/node * num_nodes
# export MASTER_PORT=64723
# export WORLD_SIZE=8

# ### get the first node name as master address - customized for vgg slurm
# ### e.g. master(gnodee[2-5],gnoded1) == gnodee2
# echo "NODELIST="${SLURM_NODELIST}
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

# ### init virtual environment if needed
# # source ~/anaconda3/etc/profile.d/conda.sh
# # conda activate myenv

# ### the command to run
# srun python main.py --net resnet18 \
# --lr 1e-3 --epochs 50 --other_args