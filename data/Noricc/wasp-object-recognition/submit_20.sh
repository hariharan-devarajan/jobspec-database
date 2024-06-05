#!/bin/bash

#SBATCH -n 20
#SBATCH -N 1 # force all cores on one node
#SBATCH -t 04:00:00
#SBATCH -J retrain
#SBATCH -o retrain_%j.out
#SBATCH -e retrain_%j.err

#SBATCH -A computehpc

cat $0 # put the script in the output file

module purge

module load  GCC/7.3.0-2.30  CUDA/9.2.88  OpenMPI/3.1.1
module load Python/3.6.6

module list

ARGS=$@

# bind threads to cores
export OMP_SCHEDULE=static
export OMP_PLACES=cores
export OMP_PROC_BIND=CLOSE

source venv/bin/activate

cd src
python retrain.py 25 1
