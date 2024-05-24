#!/bin/bash

#SBATCH --job-name=FT-Blenderbot-Small-ESCOV
#SBATCH --nodes=1
#SBATCH --account=glucas_540
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1        # number of GPUs reserved per node (here 4, or all the GPUs)
#SBATCH --cpus-per-task=10   # number of cores per task (4x10 = 40 cores, or all the cores)
#SBATCH --time=30:00:00
#SBATCH --exclusive
#SBATCH --mem=20G
#SBATCH --output=slogs/%j.%x.info.log
#SBATCH --error=slogs/%j.%x.error.log


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=2


# setup
module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate /project/glucas_540/briank/kemi-env

# set env for torch run
export MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export LD_LIBRARY_PATH=/spack/conda/miniconda3/23.3.1/lib/:$LD_LIBRARY_PATH
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# GPU Spec
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# run
echo ""
echo "Running Baseline Experiment"
echo ""

# wandb information
source "$PWD/_wandb.sh"


job_id=$SLURM_JOB_ID
experiment_nm=${job_id}.${SLURM_JOB_NAME}
python -m torch.distributed.launch --nproc_per_node 1 \
    train.py \
    --config_name emp_dialog_t5 \
    --inputter_name  emp_dialog_t5 \
    --data_name esconv \
    --knowledge_name sbert \
    --eval_input_file ./_reformat/ \
    --seed 13 \
    --max_input_length 256 \
    --max_decoder_input_length 40 \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 6 \
    --warmup_steps 100 \
    --fp16 false \
    --loss_scale 0.0 \
    --pbar true

echo "DONE"
