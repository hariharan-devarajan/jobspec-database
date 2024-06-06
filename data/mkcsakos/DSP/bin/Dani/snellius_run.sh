#!/bin/bash
#SBATCH --partition=rome
#SBATCH --time=24:00:00
#SBATCH -o seg_4_bili_train4_%j_out.txt
#SBATCH -e seg_4_bili_train4_%j_error.txt

echo "gpus $SLURM_GPUS on node: $SLURM_GPUS_ON_NODE"
echo "nodes nnodes: $SLURM_NNODES, nodeid: $SLURM_NODEID, nodelist $SLURM_NODELIST"
echo "cpus on node: $SLURM_CPUS_ON_NODE per gpu $SLURM_CPUS_PER_GPU per task $SLURM_CPUS_PER_TASK omp num thread $OMP_NUM_THREADS"
echo "tasks per node $SLURM_TASKS_PER_NODE pid $SLURM_TASK_PID"

cd /home/dszabo/DSP/bin/Dani
module load 2022
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

pip3 install zstandard
python3 /home/dszabo/DSP/bin/Dani/filter_compressed_data.py
