#!/bin/bash
#SBATCH --job-name=llama-cpp-label-mixtral
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=label-mixtral-%j.out
#SBATCH --error=label-mixtral-%j.err

HOSTNAME=$(hostname)

if [ "$HOSTNAME" == "vega.iiia.csic.es" ]
then
    spack load --first py-pandas
elif [ "$HOSTNAME" == "login*" ]
then
    module load pandas
fi

cmd=label-mixtral-$SLURM_JOB_ID.cmd
srun python3 label.py --model "models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf" --cmd $cmd
bash $cmd
