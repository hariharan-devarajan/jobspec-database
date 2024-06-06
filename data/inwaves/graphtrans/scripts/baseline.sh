#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#!
#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J baseline-generalisation-experiments
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A KRUEGER-SL3-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 3 cpus per GPU.
#SBATCH --gres=gpu:4
#! How much wallclock time will be required?
#SBATCH --time=02:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#SBATCH --output=slurm-out/%x.%j.out


#! Do not change:
#SBATCH -p ampere
#/bin/bash

python main.py --configs configs/NCI1/gcn/base.yml
python main.py --configs configs/NCI1/gcn-virtual/base.yml
python main.py --configs configs/NCI1/gnn-transformer/no-virtual/gd=128+gdp=0.1+tdp=0.1+l=3+cosine.yml
python main.py --configs configs/NCI1/gnn-transformer/no-virtual/gin+gdp=0.1+tdp=0.1+l=4+cosine.yml
python main.py --configs configs/NCI1/transformer-gnn/no-virtual/gd=128+gdp=0.1+tdp=0.1+l=3+cosine.yml
python main.py --configs configs/NCI1/transformer-gnn/no-virtual/gin+gdp=0.1+tdp=0.1+l=4+cosine.yml
python main.py --configs configs/NCI1/transformer/pooling=cls.yml


python main.py --configs configs/NCI109/gcn/base.yml
python main.py --configs configs/NCI109/gcn-virtual/base.yml
python main.py --configs configs/NCI109/gnn-transformer/no-virtual/gd=128+gdp=0.1+tdp=0.1+l=3+cosine.yml
python main.py --configs configs/NCI109/gnn-transformer/no-virtual/gin+gdp=0.1+tdp=0.1+l=4+cosine.yml
python main.py --configs configs/NCI109/transformer-gnn/no-virtual/gd=128+gdp=0.1+tdp=0.1+l=3+cosine.yml
python main.py --configs configs/NCI109/transformer-gnn/no-virtual/gin+gdp=0.1+tdp=0.1+l=4+cosine.yml
python main.py --configs configs/NCI109/transformer/pooling=cls.yml

