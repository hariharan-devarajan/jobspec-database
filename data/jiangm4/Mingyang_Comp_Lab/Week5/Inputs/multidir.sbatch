#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=ParallelRun 
#SBATCH --output=Multidir.out

module purge
module load gromacs/openmpi/intel/2018.3

mpirun -np 3 gmx_mpi mdrun -s adp -multidir T300/ T363/ T440/ -deffnm adp_exchange3temps -replex 50

echo "10" | gmx_mpi energy -f T300/adp_exchange3temps.edr -o T300_potential_energy.xvg
echo "10" | gmx_mpi energy -f T363/adp_exchange3temps.edr -o T363_potential_energy.xvg
echo "10" | gmx_mpi energy -f T440/adp_exchange3temps.edr -o T440_potential_energy.xvg
