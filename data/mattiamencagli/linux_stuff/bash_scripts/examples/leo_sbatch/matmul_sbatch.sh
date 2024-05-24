#!/bin/bash
#SBATCH --account cin_staff
#SBATCH --partition boost_usr_prod
#######SBATCH --qos boost_qos_dbg
#SBATCH --time 00:10:00            # format: HH:MM:SS
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4       # n tasks out of 32
#SBATCH --cpus-per-task=8    # 4 virtual cpu per task. So I use n physical core
#SBATCH --gres=gpu:4               # 1 gpus per node out of 4
#SBATCH --job-name=test_nsys
#SBATCH --mail-type=NONE
#SBATCH --mail-user=m.mencagli@cineca.it

## load needed modules on the node
#source ${HOME}/modules_files/pluto_mod 23.11

ml 

## print GPUs node information
nvidia-smi

cd ${HOME}/programming/cuda/matmul_multinode
pwd

echo "compilazione"
make -j -B


echo "RUN SU 2 NODI normale"
mpirun -np 4 ./matvec 1024 1024 1024 1024 1 -v


echo "RUN SU 2 NODI con nsys"
nsys profile --trace=cuda,mpi -f true -o report_prova_nsys_multinodo_8 mpirun -np 4 ./matvec 1024 1024 1024 1024 1 -v

