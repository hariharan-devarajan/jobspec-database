#!/bin/bash
#SBATCH --nodes 1
#SBATCH --mem 12G
#SBATCH --partition shortgpgpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 0-01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --qos gpgpumse
#SBATCH -o  /home/dalmiapriyam/bpp/slurmoutput/slurm-%j.out
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=dalmiap@student.unimelb.edu.au
#module purge
#module load gcc/8.3.0 fosscuda/2019b 
#module load tensorflow/2.3.1-python-3.7.4 tensorflow-probability/0.9.0-python-3.7.4
#
module load pytorch/1.5.1-python-3.7.4

python3 trainers/tprey.py
