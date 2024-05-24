#!/bin/bash 
#SBATCH --job-name=cnnf
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=30G
#SBATCH --time=2:10:00
#SBATCH --reservation=GPU
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seth.bryant@gfz-potsdam.de
#SBATCH --output=/storage/vast-gfz-hpc-01/home/bryant/LS/10_IO/2307_super/outs/train/20231208/b/slurm.log
#SBATCH --error=/storage/vast-gfz-hpc-01/home/bryant/LS/10_IO/2307_super/outs/train/20231208/b/slurm_error.log

#INSTRUCTIONS-------
#1) setup the environment per ./README.md
#2) test the environment by commenting out the python call and running the script from the handler node (may need to chmod +x ./train_1208.sh)
#3) define the variables below
#4) send to the sceduler: sbatch ./train_1208.sh (NOTE: best to run a small job firsrt)

#RUN VARS-------
base_dir=/storage/vast-gfz-hpc-01/home/bryant/LS/10_IO/2307_super/outs/train_data/20231208
out_dir=/storage/vast-gfz-hpc-01/home/bryant/LS/10_IO/2307_super/outs/train/20231208/b

#ENVIRONMENT---------
# add untested gpu-software stack
#module use /cluster/spack/2022b/share/spack/modules/linux-almalinux8-icelake
# load environment modules for using pytorch from untested gpu-software stack
#source /cluster/spack/testing/pytorch

#activate conda environment 
source /storage/vast-gfz-hpc-01/home/bryant/LS/09_REPOS/04_TOOLS/SRCNN-flood/env/conda_activate.sh



#PROGRAM--------------
echo executing
cd ..
python -O cnnf/train.py --input-data-fp "${base_dir}/train_04_p160_input_20231208.h5" --eval-data-fp "/storage/vast-gfz-hpc-01/home/bryant/LS/10_IO/2307_super/tests/train_target_1_20231208.h5" --out-dir ${out_dir} --batch-size 20 --num-epochs 100 --num-workers 6  --seed 123

#TESTING---------
#NOTE: better to use ./train_test.sh
#python -c 'import torch; print(torch.cuda.is_available())'
#python -c "import psutil; print(f'psutil version: {psutil.__version__}')"

echo finished