#!/bin/bash
## START OF HEADER ## 
#PBS -N skyline-extraction-training
#PBS -l walltime=12:00:00 
#PBS -l nodes=1
#PBS -l procs=40
#PBS -l gpus=1
#PBS -l nodes=ssd
#PBS -o skyline-extraction-training.out
#PBS -e skyline-extraction-training.err
#PBS -j oe
#PBS -V 
#PBS -m abe
#PBS M kartshy@gmail.com
## END OF HEADER ## 
cd $PBS_O_WORKDIR

## START OF TUNER ## 
wget --no-check-certificate https://www.dropbox.com/s/iaivd8wujhctpl5/cresta.tar.gz
tar -xvf cresta.tar.gz
export PATH=$PATH:$(pwd)/cresta
wget --no-check-certificate https://www.dropbox.com/s/f2uzydgmoiaue8r/skyline-extraction-training_job_20200718214228.tune
tune skyline-extraction-training_job_20200718214228.tune
## END OF TUNER ## 

singularity exec docker:modakopt/modak:tensorflow-2.1-gpu-src python3 python3 resnet/resnet_imagenet_main.py --data_dir=/mnt/nfs/home/karthee/AI/data/tf_records/train/ -batch_size=96 --train_epochs=3 --train_steps=10 --use_synthetic_data=false