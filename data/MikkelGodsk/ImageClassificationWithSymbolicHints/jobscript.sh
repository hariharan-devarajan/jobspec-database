#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J ImageNet_XGBoost
### -- ask for number of cores (default: 1) --  (prev: 6)
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"

### -- Starting time hh:mm  (seems to be working) --
##BSUB -b 20:30

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory (prev: 64GB)
#BSUB -R "rusage[mem=10GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

#BSUB -u s184399@dtu.dk
#BSUB -N

# -- end of LSF options --

##nvidia-smi

#module load python3/3.9.11
#module load numpy/1.22.3-python-3.9.11-openblas-0.3.19
#module load cuda/11.3
#module load cudnn/v8.2.0.53-prod-cuda-11.3
#source ../torch/bin/activate

# Activate conda and our conda environment:
source $HOME/miniconda3/bin/activate
conda activate Image_classification_with_symbolic_hints

export CUDA_LAUNCH_BLOCKING=1

python3 src/experiments/main.py --dataset "ImageNet" --n_workers 1 --xgb_subsample 1
#python3 tune_BERT_hparams.py
#python3 tune_svm.py
#python3 tune_bayes_fusion.py
#python3 tune_logistic_regression_based_model.py
#python3 temperature_scaling_experiment.py
