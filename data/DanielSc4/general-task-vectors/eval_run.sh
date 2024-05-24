#!/bin/bash
#SBATCH --job-name=general_task_vector_eval
#SBATCH --time=08:00:00
#SBATCH --mem=40GB
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=/home1/p313544/slurm_logs/%x.%j.out


# single GPU only script
module purge
# module load Python/3.9.6-GCCcore-11.2.0
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

echo "Python version: $(python --version)"
echo $HF_HOME
nvidia-smi
pwd

# User's vars
# All scripts must be in the PATH_TO_PRJ/scripts directory!
PATH_TO_PRJ=/home1/p313544/Documents/general-task-vectors

# checkpoint save path
export PATH_TO_STORAGE=/scratch/p313544/storage_cache/

cd $PATH_TO_PRJ
source .venv/bin/activate


echo "Executing python script..."


# 30 min each
echo "Running eval"
echo "eval on cona-facts, aggregating last-last"
declare -a aggregators=("last-last" "last-mean" "mean-last" "mean-mean")
for aggregator in "${aggregators[@]}"
do
    echo "eval on cona-facts, aggregating $aggregator"
    python -m eval \
        --model_name stabilityai/stablelm-2-zephyr-1_6b \
        --evaluator_model_name meta-llama/LlamaGuard-7b \
        --dataset_name cona-facts \
        --activation_name icl_activations_icl5_tok100.pt \
        --cie_name atp_icl5_tok100.pt \
        --technique atp \
        --aggregator $aggregator \
        --top_n_heads 10 \
        --eval_dim 110 \

done


for aggregator in "${aggregators[@]}"
do
    echo "eval on truthful_eval, aggregating $aggregator"
    python -m eval \
        --model_name stabilityai/stablelm-2-zephyr-1_6b \
        --evaluator_model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --dataset_name truthful_eval \
        --activation_name icl_activations_icl5_tok100.pt \
        --cie_name atp_icl5_tok100.pt \
        --technique atp \
        --aggregator $aggregator \
        --top_n_heads 10 \
        --eval_dim 110 \

done

for aggregator in "${aggregators[@]}"
do
    echo "eval on untruthful_eval, aggregating $aggregator"
    python -m eval \
        --model_name stabilityai/stablelm-2-zephyr-1_6b \
        --evaluator_model_name mistralai/Mistral-7B-Instruct-v0.2 \
        --dataset_name untruthful_eval \
        --activation_name icl_activations_icl5_tok100.pt \
        --cie_name atp_icl5_tok100.pt \
        --technique atp \
        --aggregator $aggregator \
        --top_n_heads 10 \
        --eval_dim 110 \

done



# performance notes on A100 40GB
#   1B model (float32) + 7B eval model (8bit) = 20m, eval_dim=110
