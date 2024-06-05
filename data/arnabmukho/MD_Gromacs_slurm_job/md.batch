#!/bin/bash
#
#SBATCH --partition=xyz         # Partition 
#SBATCH --job-name=gromacs_md
#SBATCH --time=3-00:00:00       # WallTime: set according to the server allotment

#SBATCH --nodes           1    # May vary
#SBATCH --ntasks-per-core 40    # Bind 1 MPI tasks to 1 CPU core
#SBATCH --ntasks-per-node 40  # Should be less/equal to the number of CPU cores
#SBATCH --cpus-per-task   2    # Should be 2, unless you have a better guess
#SBATCH --gres=gpu:2    # Should be 2, or better*

#SBATCH -o slurm.%j.out        # output
#SBATCH -e slurm.%j.err        # errors

module purge
module load  apps/gromacs/2020.3/mpich+gcc_10.2.0/gpu        # Depends on availble module of gromacs in the server

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export UCX_NET_DEVICES=mlx5_0:1

cd $SLURM_SUBMIT_DIR

mpirun -np 40 gmx_mpi grompp -f minimization.mdp -c input.gro -r input.gro -p topol.top -o em.tpr && gmx_mpi mdrun -deffnm em && gmx_mpi grompp -f  equilibration.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -n index.ndx && gmx_mpi mdrun -deffnm nvt && gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -n index.ndx && gmx_mpi mdrun -deffnm npt && gmx_mpi grompp -f production.mdp -c npt.gro -t npt.cpt -p topol.top -o md_new.tpr -n index.ndx && gmx_mpi mdrun -deffnm md_new
