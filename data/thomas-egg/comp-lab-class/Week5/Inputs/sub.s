#!/bin/bash
#SBATCH --nodes=1
#SBATCH	--ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH	--time=24:00:00
#SBATCH --mem=3GB
#SBATCH --mail-type=END
#SBATCH --mail-user=tje3676@nyu.edu
#SBATCH --job-name=replica-exchange

module purge
module load gromacs/openmpi/intel/2018.3

# Run gromacs
srun --pty /bin/bash 
mpirun -np 3 gmx_mpi mdrun -s adp -multidir T300/ T363 T440/ -deffnm adp_exchange3temps -replex 50
