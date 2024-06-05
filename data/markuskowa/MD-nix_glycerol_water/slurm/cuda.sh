#!/usr/bin/env nix-shell
#!nix-shell -i bash /cfs/home/fpera/gmx_simulations/nix_files/cuda.nix

# Number of tasks
#SBATCH -n 24

# Number of threads per task
#SBATCH -c 1

# requested number of GPUs
#SBATCH --gres gpu:a100:4
#SBATCH -p ampere
#SBATCH -J gromacs
#SBATCH -t2-00:00  # time limit: (D-HH:MM) 

# Adjust line #2 to the absolute path of your cuda.nix file.

# Best load is 6 tasks per GPU
# e.g.: -n6 --gres gpu:a100:1

## Remember the total number of cores = 2 x 24 per GPU (Ampere) so that
## n x c = 48 (running on a single node)

# Automatic selection of ntomp argument based on "-c" argument to sbatch
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
   ntomp="$SLURM_CPUS_PER_TASK"
else
   ntomp="1"
fi

# Make sure to set OMP_NUM_THREADS equal to the value used for ntomp
# to avoid complaints from GROMACS
export OMP_NUM_THREADS=$ntomp

# for single task
# gmx mdrun -ntomp $ntomp -deffnm em

# for multi task
mpirun gmx_mpi mdrun -ntomp $ntomp -deffnm $1
