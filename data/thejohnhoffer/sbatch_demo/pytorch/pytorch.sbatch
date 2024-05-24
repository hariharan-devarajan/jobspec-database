#!/bin/bash
#
# add all other SBATCH directives here...
#
#SBATCH -p cox # Has coxgpu[1-6]
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=8000
#SBATCH -t 3-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=you@example.com

# Source your python instance
source new-modules.sh
module load python/2.7.11-fasrc01
source activate $VENV_NAME

# we are working
RUNNING="python -u $PYTHON $SLURM_ARRAY_TASK_ID"
echo $RUNNING
time $RUNNING

# end of program
exit 0;
