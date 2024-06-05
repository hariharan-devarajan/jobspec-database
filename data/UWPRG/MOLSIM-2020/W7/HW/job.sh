#!/bin/bash
## Job Name
#SBATCH --job-name=molsim_hw3
## Allocation Definition
#SBATCH --account=pfaendtner
#SBATCH --partition=pfaendtner
## Resources
## Nodes
#SBATCH --nodes=1
## Tasks per node
#SBATCH --ntasks=2
## Walltime
#SBATCH --time=00:30:00
# E-mail Notification, see man sbatch for options
#SBATCH --mail-type=NONE
## Memory per node
#SBATCH --mem=22G

# %% The lines below indicate the path to GROMACS and PLUMED; this path will depend on simulation package in use %%
 
# module load icc_15.0.3-impi_5.0.3
# source /gscratch/pfaendtner/cdf6gc/codes/GROMACS/activate_gmx2016.sh

# walkers, Nodes, MPI ranks/walker, OpenMP threads/rank
# |1 |1 |1 |1 |
# |1 |1 |2 |1 |
# |1 |1 |4 |1 |
# |1 |1 |8 |1 |
# |1 |1 |16 |1 |
# |1 |1 |8 |2 |
# |1 |1 |4 |4 |
# |1 |1 |2 |8 |
# |1 |1 |1 |16 |
# |2 |1 |8 |1 |
# |4 |1 |4 |1 |
# |8 |1 |2 |1 |
# |2 |2 |16 |1 |
# |4 |2 |8 |1 |
# |8 |2 |4 |1 |


# %% Command that calls on the GROMACS MD machine to run your simulation. Again, this may be different for different projects %%
# mpiexec.hydra -n 1 gmx_mpi mdrun -deffnm npt -ntomp 1 &> log.txt


cat conditions.txt | while read line; do
  
  echo $line
done
