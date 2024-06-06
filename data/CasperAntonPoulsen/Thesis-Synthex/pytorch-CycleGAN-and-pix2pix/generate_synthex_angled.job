#!/bin/sh

### General options
### –- specify queue --
#BSUB -q p1
### -- set the job Name --
#BSUB -J generate_synthex_images
### -- ask for number of cores (default: 1) --
#BSUB -n 16
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=11800MB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u caap@itu.dk
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o /dtu/p1/johlau/Thesis-Synthex/jobs/lsf10/logs/R-generate_synthex_images_%J.out
#BSUB -e /dtu/p1/johlau/Thesis-Synthex/jobs/lsf10/logs/R-generate_synthex_images_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/12.3.2
module load cudnn/v8.9.7.29-prod-cuda-12.X 


/dtu/p1/johlau/Thesis-Synthex/.venv/bin/python3 /dtu/p1/johlau/Thesis-Synthex/pytorch-CycleGAN-and-pix2pix/test.py \
    --dataroot /dtu/p1/johlau/Thesis-Synthex/data/RAD-ChestCT-DRR-angled-CROPPED \
    --name synthex_512 \
    --model test \
    --load_size 512 \
    --crop_size 512 \
    --preprocess none \
    --results_dir /dtu/p1/johlau/Thesis-Synthex/data/RAD-ChestCT-Synthex-angled \
    --checkpoints_dir /dtu/p1/johlau/Thesis-Synthex/synthex/data/pytorch_model \
    --dataset_mode single \
    --netG resnet_9blocks \
    --netD basic \
    --no_dropout \
    --norm instance \
    --no_flip \
    --direction AtoB \
    --input_nc 1 \
    --output_nc 1 \
    --num_test -1