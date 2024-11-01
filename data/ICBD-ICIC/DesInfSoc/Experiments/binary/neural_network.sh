#!/bin/bash
#SBATCH --job-name=itrust-neural_network
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=4
#SBATCH --array=34,37,39,40,44,48
#SBATCH --time=07-00:00:00
#SBATCH --output=outputs/neural_network-%A-%a.out
#SBATCH --error=errors/neural_network-%A-%a.err

source ../.experiments_env/bin/activate

srun python neural_network.py ${SLURM_ARRAY_TASK_ID} context2_ONLY-ACTION-SPREAD20_K3_H4_P12-BINARY
