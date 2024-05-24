#!/bin/bash

dirs=(
    # "codellama-7b"
    # "codellama-13b"
    # "codellama-33b"
    # "deepseek-base-1.3b"
    # "deepseek-base-6.7b"
    # "deepseek-base-33b"
    # "deepseek-instruct-1.3b"
    # "deepseek-instruct-6.7b"
    # "deepseek-instruct-33b"
    # "magicoder-ds-6.7b"
    # "magicoder-cl-7b"
    # "mistral-7b"
    # "mixtral-8x7b"
    # "starcoder-16b"
    "wizard-13b"
    "wizard-34b"
)

models=(
    # "codellama/CodeLlama-7b-hf"
    # "codellama/CodeLlama-13b-hf"
    # "codellama/CodeLlama-34b-hf"
    # "deepseek-ai/deepseek-coder-1.3b-base"
    # "deepseek-ai/deepseek-coder-6.7b-base"
    # "deepseek-ai/deepseek-coder-33b-base"
    # "deepseek-ai/deepseek-coder-1.3b-instruct"
    # "deepseek-ai/deepseek-coder-6.7b-instruct"
    # "deepseek-ai/deepseek-coder-33b-instruct"
    # "ise-uiuc/Magicoder-S-DS-6.7B"
    # "ise-uiuc/Magicoder-S-CL-7B"
    # "mistralai/Mistral-7B-v0.1"
    # "mistralai/Mixtral-8x7B-v0.1"
    # "bigcode/starcoder"
    "WizardLM/WizardCoder-Python-13B-V1.0"
    "WizardLM/WizardCoder-Python-34B-V1.0"
)

temperatures=(0.2 0.8)

for ((i=0; i<${#models[@]}; i++)); do
    model=${models[$i]}
    base_dir=${dirs[$i]}
    if [ "$model" == "mistralai/Mixtral-8x7B-v0.1" ]; then
        tp_size=2
    else
        tp_size=1
    fi
    echo "Model $model, TP Size $tp_size"

    for temperature in "${temperatures[@]}"; do
        dir="${base_dir}_temp${temperature}_output"
        cat <<EOF > temp_sbatch_script.sh
#!/bin/bash
#SBATCH --output=/fsx-codegen/gua/Documents/cruxeval/slurm_logs/slurm-%A-%a.out
#SBATCH --error=/fsx-codegen/gua/Documents/cruxeval/slurm_logs/slurm-%A-%a.err
#SBATCH --partition=learnlab
#SBATCH --array=0-19
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=0GB
#SBATCH --time=03:00:00

dir=$dir
SIZE=479
GPUS=20

i=\$SLURM_ARRAY_TASK_ID
ip=\$((\$i+1))

echo \$dir
mkdir -p model_generations_raw/\$dir

string="Starting iteration \$i with start and end  \$((\$i*SIZE/GPUS)) \$((\$ip*SIZE/GPUS))"
echo \$string

python main.py \
    --model $model \
    --use_auth_token \
    --trust_remote_code \
    --tasks output_prediction \
    --batch_size 10 \
    --n_samples 10 \
    --max_length_generation 1024 \
    --precision bf16 \
    --limit \$SIZE \
    --temperature $temperature \
    --save_generations \
    --save_generations_path model_generations_raw/\${dir}/shard_\$((\$i)).json \
    --start \$((\$i*SIZE/GPUS)) \
    --end \$((\$ip*SIZE/GPUS)) \
    --shuffle \
    --tensor_parallel_size $tp_size
EOF
        sbatch temp_sbatch_script.sh
        rm temp_sbatch_script.sh
    done
done