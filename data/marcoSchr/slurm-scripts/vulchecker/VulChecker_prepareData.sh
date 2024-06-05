#!/bin/bash
#
#SBATCH --job-name=vulchecker_prepare_data
#SBATCH --output=/ukp-storage-1/schroeder_e/vulchecker_prepare_data.txt
#SBATCH --mail-user=marco.schroeder@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --account=tk-student
#SBATCH --partition=gpu-tk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/schroeder_e/VulChecker/venv/bin/activate
module load cuda/10.0
cd /ukp-storage-1/schroeder_e/VulChecker

export PATH=/ukp-storage-1/schroeder_e/llvm/llvm-project/clang-build/bin:$PATH
export PATH=/ukp-storage-1/schroeder_e/llvm/llvm-project/llvm-build/bin:$PATH
export PATH=/ukp-storage-1/schroeder_e/ninja:$PATH

cd data/
hector configure --llap-lib-dir /ukp-storage-1/schroeder_e/llvm/llvm-project/llvm-build/lib --labels labels.json cmake "" 121 190 415 416
