#!/bin/bash
# JOB HEADERS HERE

#SBATCH --job-name=run-gromace
#SBATCH --nodes=1
#SBATCH --tasks-per-node=48
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
module purge
module load gromacs/openmpi/intel/2020.4
#grep -v HOH 1aki.pdb > 1AKI_clean.pdb
#mpirun -np 1 gmx_mpi pdb2gmx -f 1AKI_clean.pdb -o 1AKI_processed.gro -water spce
#15
#mpirun -np 1 gmx_mpi editconf -f 1AKI_processed.gro -o 1AKI_newbox.gro -c -d 1.0 -bt cubic
#mpirun -np 1 gmx_mpi solvate -cp 1AKI_newbox.gro -cs spc216.gro -o 1AKI_solv.gro -p topol.top
#mpirun -np 1 gmx_mpi grompp -f ions.mdp -c 1AKI_solv.gro -p topol.top -o ions.tpr
#mpirun -np 1 gmx_mpi genion -s ions.tpr -o 1AKI_solv_ions.gro -p topol.top -pname NA -nname CL -neutral
#13
mpirun -np 1 gmx_mpi grompp -f minim.mdp -c 1AKI_solv_ions.gro -p topol.top -o 1AKI_solv_ions_em.tpr
mpirun -np 1 gmx_mpi mdrun -s 1AKI_solv_ions_em.tpr -deffnm 1AKI_solv_ions_em
mpirun -np 1 gmx_mpi grompp -f nvt.mdp -c 1AKI_solv_ions_em.gro -r 1AKI_solv_ions_em.gro -p topol.top -o nvt.tpr
mpirun gmx_mpi mdrun -deffnm nvt
mpirun -np 1 gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
mpirun gmx_mpi mdrun -deffnm npt
mpirun -np 1 gmx_mpi grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_1.tpr
mpirun gmx_mpi mdrun -deffnm md_0_1
#mpirun -np 1 gmx_mpi grompp -f md_50.mdp -c npt.gro -t npt.cpt -p topol.top -o md_0_50_48tasks.tpr
#mpirun gmx_mpi mdrun -deffnm md_0_50_48tasks