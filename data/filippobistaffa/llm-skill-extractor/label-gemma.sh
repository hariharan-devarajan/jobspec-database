#!/bin/bash
#SBATCH --job-name=llama-cpp-label-gemma
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --output=label-gemma-%j.out
#SBATCH --error=label-gemma-%j.err

HOSTNAME=$(hostname)

if [ "$HOSTNAME" == "vega.iiia.csic.es" ]
then
    spack load --first py-pandas
elif [ "$HOSTNAME" == "login*" ]
then
    module load pandas
fi

cmd=label-gemma-$SLURM_JOB_ID.cmd
srun python3 label.py --model "models/gemma-7b-it-Q4_K_M.gguf" --cmd $cmd
bash $cmd
