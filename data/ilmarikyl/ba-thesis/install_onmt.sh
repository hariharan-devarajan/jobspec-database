#!/bin/bash -l
# created: Sept, 2018
# author: vazquezc
#SBATCH -J onmt_install
#SBATCH -o out_%J.onmt_install.txt
#SBATCH -e err_%J.onmt_install.txt
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH --mem-per-cpu=1g
#SBATCH --mail-type=NONE
#SBATCH --mail-user=raul.vazquez@helsinki.fi
#SBATCH --gres=gpu:p100:1
module purge
module load python-env/intelpython3.6-2018.3 gcc/5.4.0 cuda/9.0 cudnn/7.1-cuda9
module list 
mkdir -p /wrk/${USER}/git/
cd /wrk/${USER}/git/
if [ ! -f "./OpenNMT-py/README.md" ]; then
    echo "cloning OpenNMT-py repository"
    git clone --recursive https://github.com/OpenNMT/OpenNMT-py.git
    # OR our branch:
    # git clone --recursive git@github.com:Helsinki-NLP/OpenNMT-py.git
    cd OpenNMT-py
  else
      echo "repository already exists"
      cd OpenNMT-py
      echo "pulling repository"
      git pull origin master 
fi
pip install git+https://github.com/pytorch/text --user