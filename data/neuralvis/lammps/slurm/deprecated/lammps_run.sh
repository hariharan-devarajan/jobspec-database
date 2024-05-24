#!/bin/bash --login

###
# job name
#SBATCH --job-name=lmp_diffuse_2d
# job stdout file
#SBATCH --output=lmp_diffuse_2d.%J.out
# job stderr file
#SBATCH --error=lmp_diffuse_2d.%J.err
# On failure, requeue for another try
#SBATCH --requeue
# maximum job time in HH:MM:SS
#SBATCH --time=06:00:00
#SBATCH --nodes=5
# maximum memory
#SBATCH --mem-per-cpu=1024
# run a single task
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
###


module load cce/10.0.3
module load craype/2.7.2
module load cray-mpich/8.0.15
module load cray-libsci/20.08.1.2

cd /home/users/msrinivasa/develop/lammps
srun build/lmp+pat -i examples/DIFFUSE/in.msd.2d

