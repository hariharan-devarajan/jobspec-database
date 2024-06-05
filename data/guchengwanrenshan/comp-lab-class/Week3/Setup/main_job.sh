#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=run-gromacs
#SBATCH --nodes=1
#SBATCH --tasks-per-node=48
#SBATCH --mem=16GB
#SBATCH --time=14:00:00
##SBATCH --gres=gpu:1 
module purge
module load gromacs/openmpi/intel/2020.4

mpirun gmx_mpi rms -s em.tpr -f md_0_1_noPBC.xtc -o rmsd_xtal.xvg -tu ns
mpirun gmx_mpi editconf -f 1AKI_processed.gro -o 1AKI_newbox.gro -c -d 1.0 -bt cubic
mpirun gmx_mpi solvate -cp 1AKI_newbox.gro -cs spc216.gro -o 1AKI_solv.gro -p topol.top
mpirun gmx_mpi grompp -f ions.mdp -c 1AKI_solv.gro -p topol.top -o ions.tpr
mpirun gmx_mpi genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral
mpirun gmx_mpi grompp -f minim.mdp -c 1AKI_solv_ions.gro -p topol.top -o em.tpr
mpirun gmx_mpi mdrun -v -deffnm em
mpirun gmx_mpi grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
mpirun gmx_mpi mdrun -deffnm nvt
mpirun gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
mpirun gmx_mpi  mdrun -deffnm npt
mpirun gmx_mpi  grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr
mpirun gmx_mpi  mdrun -deffnm md_0_1
#mpirun gmx_mpi  mdrun -deffnm md_0_1 -nb gpu"
