#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J sprites
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
cd /work3/danli/

# Load modules
module purge 
module load gcc/7.5.0
module load cuda/11.6
module load matplotlib/3.6.0-numpy-1.23.3-python-3.8.14
module load pandas/1.4.4-python-3.8.14
module load cython/0.29.32-python-3.8.14
module load scipy/1.9.1-python-3.8.14

# env 
source adlcv2/bin/activate

# run detr model first  
cd CubeDETR_3D/detr/
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path ../datasets/COCO/  --epochs 1

/zhome/db/7/181395/miniconda3/envs/cubercnn/bin/python \
    demo/demo.py --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
    --input-folder "datasets/coco_examples" --threshold 0.25 --display \ 
    MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth OUTPUT_DIR output/demo