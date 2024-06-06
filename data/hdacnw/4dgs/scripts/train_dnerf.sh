#!/bin/bash
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --time 4:00:00
#SBATCH --mem=32G

#SBATCH --output=./output/4dgs_out.txt
#SBATCH --error=./output/4dgs_error.txt

eval "$(conda shell.zsh hook)"

module load CUDA/11.8.0
module load cudnn/v8.8.0
nvidia-smi

conda activate threestudio

export CUDA_HOME="/opt/common/cuda/cuda-11.8.0/"
export LD_LIBRARY_PATH="/opt/common/cudnn/cudnn-11.x-8.8.0.121/lib64:/opt/common/cuda/cuda-11.8.0/lib64"

# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/penguin_static_16 --port 7860 --expname "penguin_static_16" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/penguin_static_4 --port 7860 --expname "penguin_static_4" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/penguin_dynamic_16 --port 7860 --expname "penguin_dynamic_16" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/penguin_dynamic_4 --port 7860 --expname "penguin_dynamic_4" --configs arguments/dnerf/chicken_cam.py

python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/rabbit_static_16 --port 7860 --expname "rabbit_static_16" --configs arguments/dnerf/chicken_cam.py
python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/rabbit_static_4 --port 7860 --expname "rabbit_static_4" --configs arguments/dnerf/chicken_cam.py
python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/rabbit_dynamic_16 --port 7860 --expname "rabbit_dynamic_16" --configs arguments/dnerf/chicken_cam.py
python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/rabbit_dynamic_4 --port 7860 --expname "rabbit_dynamic_4" --configs arguments/dnerf/chicken_cam.py

# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/camel_static_16 --port 7860 --expname "camel_static_16" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/camel_static_4 --port 7860 --expname "camel_static_4" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/camel_dynamic_16 --port 7860 --expname "camel_dynamic_16" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/camel_dynamic_4 --port 7860 --expname "camel_dynamic_4" --configs arguments/dnerf/chicken_cam.py

# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/camel_static_16 --port 7860 --expname "chicken_static_16" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/camel_static_4 --port 7860 --expname "chicken_static_4" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/camel_dynamic_16 --port 7860 --expname "chicken_dynamic_16" --configs arguments/dnerf/chicken_cam.py
# python train.py -s /fs/nexus-scratch/hwl/prj-group-animal-synthetic/data/camel_dynamic_4 --port 7860 --expname "chicken_dynamic_4" --configs arguments/dnerf/chicken_cam.py

# exp_name1=$1

# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dnerf/lego --port 6068 --expname "$exp_name1/lego" --configs arguments/$exp_name1/lego.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dnerf/bouncingballs --port 6266 --expname "$exp_name1/bouncingballs" --configs arguments/$exp_name1/bouncingballs.py &
# wait
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dnerf/jumpingjacks --port 6069 --expname "$exp_name1/jumpingjacks" --configs arguments/$exp_name1/jumpingjacks.py  &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dnerf/trex --port 6070 --expname "$exp_name1/trex" --configs arguments/$exp_name1/trex.py &
# wait
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dnerf/mutant --port 6068 --expname "$exp_name1/mutant" --configs arguments/$exp_name1/mutant.py &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dnerf/standup --port 6066 --expname "$exp_name1/standup" --configs arguments/$exp_name1/standup.py &
# wait
# export CUDA_VISIBLE_DEVICES=2&&python train.py -s data/dnerf/hook --port 6069 --expname "$exp_name1/hook" --configs arguments/$exp_name1/hook.py  &
# export CUDA_VISIBLE_DEVICES=3&&python train.py -s data/dnerf/hellwarrior --port 6070 --expname "$exp_name1/hellwarrior" --configs arguments/$exp_name1/hellwarrior.py &
# wait
echo "Done"