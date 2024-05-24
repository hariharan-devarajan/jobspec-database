#!/bin/sh

#SBATCH -J assignment_05
#SBATCH -o assignment_05_%j.out
#SBATCH -p htc
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

module purge
wget https://github.com/chiawang/DS-7347/blob/main/assignment_05/assignment_05.R

spack env activate r-base
Rscript assignment_05.R

cat /proc/cpuinfo
echo "Job ID is $SLURM_JOBID"
echo "Job Name is $SLURM_JOB_NAME"
echo "Cluster Name is $SLURM_CLUSTER_NAME"
echo "Number of nodes is $SLURM_NNODES"
echo "Memory per Node is $SLURM_MEM_PER_NODE"
echo "Memory per CPU is $SLURM_MEM_PER_CPU"
echo "Cores per Node is $SLURM_CPUS_ON_NODE"
