#!/bin/bash
#SBATCH --job-name=lammps        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=end          # send email when job ends
##SBATCH --mail-user=yp0007@princeton.edu

module purge
module load anaconda3/2021.5
conda activate deepmd

lmp -in in.lammps