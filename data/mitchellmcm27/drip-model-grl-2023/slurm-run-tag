#!/bin/bash 
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --time=18:00:00
#SBATCH --job-name=drip-model-grl-2023
#SBATCH --output=output_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mitchellmcm27@gmail.com
 
cd $SLURM_SUBMIT_DIR
 
module load NiaEnv/2019b
module load gcc/8.3.0 
module load boost/1.70.0 
module load openmpi/4.0.1 
module load dealii/9.2.0 
module load cmake/3.17.3

tag=v0

if [[ $SLURM_ARRAY_TASK_ID ]]
then
  id=$SLURM_ARRAY_TASK_ID
  tag=v${id%?}.${id: -1}
else
  tag=v0
fi

echo tag 

./run-tag $tag $HOME/aspect/build/aspect