#!/bin/bash

#SBATCH --job-name=qlora
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=36GB
#SBATCH --time=47:59:00
#SBATCH --gres=gpu:a100:1
module purge
module load cudnn/8.6.0.163-cuda11
module load cuda/11.3.1
overlay_ext3=/home/xc1490/home/apps/llm2/overlay-15GB-500K.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
export BNB_CUDA_VERSION=113
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/cuda/11.3.1/lib64

cd /home/xc1490/home/projects/hpml/project/bitsandbytes 
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.0 --save output/llama_0.0 > output/llama_0.0.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.1 --save output/llama_0.1 > output/llama_0.1.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.2 --save output/llama_0.2 > output/llama_0.2.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.3 --save output/llama_0.3 > output/llama_0.3.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.4 --save output/llama_0.4 > output/llama_0.4.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.5 --save output/llama_0.5 > output/llama_0.5.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.6 --save output/llama_0.6 > output/llama_0.6.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.7 --save output/llama_0.7 > output/llama_0.7.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.8 --save output/llama_0.8 > output/llama_0.8.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.9 --save output/llama_0.9 > output/llama_0.9.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.0 --save output/llama_0.0 > output/llama_0.0.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.19 --save output/llama_0.19 > output/llama_0.19.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.36 --save output/llama_0.36 > output/llama_0.36.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.51 --save output/llama_0.51 > output/llama_0.51.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.64 --save output/llama_0.64 > output/llama_0.64.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.75 --save output/llama_0.75 > output/llama_0.75.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.84 --save output/llama_0.84 > output/llama_0.84.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.91 --save output/llama_0.91 > output/llama_0.91.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.96 --save output/llama_0.96 > output/llama_0.96.txt
python llama.py --model huggyllama/llama-13b --dataset c4 --tokenizer huggyllama/llama-13b --sparsity 0.99 --save output/llama_0.99 > output/llama_0.99.txt"
