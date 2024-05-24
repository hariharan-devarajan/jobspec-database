#!/bin/bash
#Uso:
#   $ sbatch stream1.srm    (roda em prj)(anotar o my_job_no)
#   $ squeue --job my_job_no  (verifica jobs)
#   saída slurm-my_job_no.out em scracth
#SBATCH --nodes=1           # utilizar um único nó
#SBATCH --ntasks-per-node=1 # ntasks(max tasks) invoked on each node
#SBATCH --ntasks=1          # no total de processos MPI
#SBATCH --cpus-per-task=1   # number of threads (max 24)
#SBATCH -p cpu_dev          # Fila (partition) ate 24 cores
#SBATCH -J list06           # Nome do job
#SBATCH --time=00:02:00     # Altera o tempo limite para 2 minutos
#SBATCH --exclusive         # Utilização exclusiva dos nós durante o job

# Exibe os nós alocados para o Job
echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

# Config. compiladores p/ suite de compiladores da Intel e MPI compilado com Intel
module load intel_psxe/2019

# Configura o executável
EXEC1=/scratch/padinpe/____/stream

# exibe informações sobre o executável (opcional)
#/usr/bin/ldd $EXEC1

# Configura o número de threads conforme o parâmetro acima do SLURM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Imprime o número de threads 
echo "SLURM_CPUS_PER_TASK" $SLURM_CPUS_PER_TASK

# Executa com 1 processo e define número de threads com a opção "-c":
srun -N 1 -c $SLURM_CPUS_PER_TASK $EXEC1
