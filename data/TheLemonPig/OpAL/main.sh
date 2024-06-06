#!/bin/sh

#SBATCH -J OpAL-Star
#SBATCH --account=carney-brainstorm-condo
#SBATCH --time=2:00:00
#SBATCH --array=0-999
#SBATCH --mem=10GB
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

module load graphviz/2.40.1
module load python/3.9.0
module load git/2.29.2
source ~/OpAL/venv/bin/activate
cd /users/jhewson/OpAL/

# Run job
python main_slurm.py --slurm_id $SLURM_ARRAY_TASK_ID