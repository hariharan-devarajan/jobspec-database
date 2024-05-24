#!/bin/bash --login
#SBATCH   --job-name=RGZ_test
#SBATCH   --nodes=1
#SBATCH   --cpus-per-task=1
#SBATCH   --time=10:00:00
#SBATCH   --account=pawsey0245
#SBATCH   --partition=gpuq
#SBATCH   --gres=gpu:4

module load broadwell gcc/5.4.0 cuda
module load python numpy

# yes we are using Kevin's MKL library and software cuDDN library to launch tensorflow!
export LD_LIBRARY_PATH=/group/pawsey0245/kvinsen/tensorflow/third_party/mkl:/group/pawsey0245/software/cuda/lib64:$LD_LIBRARY_PATH

srun -n 1 /group/pawsey0245/cwu/pyml/bin/python /group/pawsey0129/cwu/rgz-faster-rcnn/tools/train_net.py --device gpu --device_id 0 --imdb rgz_2017_train22 --iters 80000 --cfg /group/pawsey0129/cwu/rgz-faster-rcnn/experiments/cfgs/faster_rcnn_end2end_test.yml --network VGGnet_train22 --weights /home/cwu/rgz-ml-ws/data/pretrained_model/VGG_imagenet.npy

#srun -n 1 /group/pawsey0129/cwu/rgz-faster-rcnn/experiments/scripts/faster_rcnn_end2end.sh gpu 0 VGGnet_train22 rgz22 600 --weights /home/cwu/rgz-ml-ws/data/pretrained_model/VGG_imagenet.npy

