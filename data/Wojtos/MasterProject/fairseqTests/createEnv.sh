#!/bin/bash
#SBATCH -e logs/log-%j.err
#SBATCH -o logs/log-%j.out

module load plgrid/tools/python-intel/3.6.2
module load plgrid/apps/cuda/10.1

virtualenv -p python $HOME/fairseqTests/venv
source $HOME/fairseqTests/venv/bin/activate

pip install fairseq
pip install fastBPE sacremoses subword_nmt

# git clone https://github.com/pytorch/fairseq.git
# cd fairseq
# pip install -r requirements.txt
# python setup.py build develop
# cd ..
