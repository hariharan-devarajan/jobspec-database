#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=48000M
#SBATCH --time=0-02:59
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-hongyanz
#SBATCH --mail-user=haochen.sun@uwaterloo.ca
#SBATCH --mail-type=ALL

deactivate
module purge
module load gcc/9.3.0 python/3.10 arrow/11.0.0 nodejs cuda/11.7 cmake protobuf cudnn scipy-stack rust
source $HOME/Quantization_ENV/bin/activate
echo "The Python used is $(which python)"

TRANSFORMERS_CACHE=./model-storage CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 4 --quant ldlq --incoh_processing #--groupsize 128
# TRANSFORMERS_CACHE=./model-storage CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 2 --quant ldlq --incoh_processing #--groupsize 128
TRANSFORMERS_CACHE=./model-storage CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 1 --quant gptq --groupsize 128
TRANSFORMERS_CACHE=./model-storage CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 1 --quant gptq --incoh_processing --groupsize 128

# TRANSFORMERS_CACHE=./model-storage CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 2 --true-sequential --act-order --groupsize 128 --eval
# Benchmark generating a 2048 token sequence with the saved model
# TRANSFORMERS_CACHE=./model-storage CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 2 --groupsize 128 --load llama7b-2bit-128g.pt --benchmark 2048 --check

# TRANSFORMERS_CACHE=./model-storage CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 1 --true-sequential --act-order --groupsize 128 --eval
# Benchmark generating a 2048 token sequence with the saved model
# TRANSFORMERS_CACHE=./model-storage CUDA_VISIBLE_DEVICES=0 python llama.py meta-llama/Llama-2-7b-hf c4 --wbits 1 --groupsize 128 --load llama7b-1bit-128g.pt --benchmark 2048 --check