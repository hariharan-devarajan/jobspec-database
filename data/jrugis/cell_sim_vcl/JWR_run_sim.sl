#!/bin/bash
#SBATCH -J cell_sim_vcl
#SBATCH -A nesi00119          # Project Account
#SBATCH --time=0:15:00       # Walltime HH:MM:SS
#SBATCH --mem-per-cpu=32G     # Memory
#SBATCH --ntasks=1            # number of tasks
#SBATCH --cpus-per-task=1     # number of threads
##SBATCH --nodes=1             # number nodes
#SBATCH -C avx                 # sb=Sandybridge wm=Westmere avx=Sandybridge or Ivybridge

# output some information
echo $HOSTNAME

# load module(s)
module load intel/2015a
module load Python/3.5.0-intel-2015a

# run the job
srun -o sim.log ./src/cell_3d

# compare the result
python scripts/compare_bin.py c.bin c_REF.bin 1e-5
