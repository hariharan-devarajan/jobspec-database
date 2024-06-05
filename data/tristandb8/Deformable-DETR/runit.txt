#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
#SBATCH --time=03:00:00
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=D-DETR_test
#SBATCH --gres=gpu:2
# Load modules
conda create -n deformable_detr python=3.7 pip
conda activate deformable_detr
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
cd Deformable-DETR
cd ./models/ops
sh ./make.sh
python test.py