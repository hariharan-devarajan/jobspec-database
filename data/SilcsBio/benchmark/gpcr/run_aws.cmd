#!/bin/bash
#SBATCH --output benchmark_test.out
#SBATCH --job-name g_202X.Y
#SBATCH --ntasks 8
#SBATCH --partition gpu
#SBATCH --time 24:00:00

nproc=8
#GMXDIR="/shared/apps/gromacs/2020.3/bin"
#GMXDIR=/shared/apps/gromacs/2021.3/bin
GMXDIR="/shared/apps/gromacs/2021.4/bin"
charmmff="charmm36.ff"

mpirun="mpirun --leave-session-attached"

mdrun="${GMXDIR}/gmx mdrun -nt $nproc"  # gmx mdrun command

# do not edit below
env
nvidia-smi
export GMX_MAXBACKUP=-1

$GMXDIR/gmx grompp -f emin.mdp -c 3pbl.pdb -p 3pbl.top -o min -r 3pbl.pdb -maxwarn 4 -n prot_lipid_silcs.ndx
$mdrun -deffnm min -c min.pdb

$GMXDIR/gmx grompp -f equil.mdp -c min.pdb -p 3pbl.top -o equil -r 3pbl.pdb -maxwarn 4 -n prot_lipid_silcs.ndx
$mdrun -deffnm equil -c equil.pdb

$GMXDIR/gmx grompp -f prod.mdp -c equil.pdb -p 3pbl.top -o prod -r 3pbl.pdb -maxwarn 4 -n prot_lipid_silcs.ndx 
$mdrun -deffnm prod -c prod.pdb

