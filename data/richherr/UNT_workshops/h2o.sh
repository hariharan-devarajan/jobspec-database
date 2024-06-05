#!/bin/bash
#SBATCH --job-name=h2o_arrayJob
#SBATCH --output=h2o_arrayJob_%A_%a.out
#SBATCH --error=h2o_arrayJob_%A_%a.err
#SBATCH -N 1
#SBATCH -p development
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-user=richherr@unt.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Load module or activate conda environment
source ~/conda.init
conda activate h2oai

# Initialize H2O cluster
srun python h2o_randomForest.py