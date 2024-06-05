#!/bin/bash
# Create a script to run 25 FFF molecules in water starting wit a lone peptide.
# The mpirun commands here were set up to run on Bridges in October 2020.

#SBATCH --mail-user=mh1314@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=BRIDGES_fff25_200
#SBATCH --partition RM
#SBATCH -N 1 --ntasks-per-node 28
#SBATCH --time=48:00:00
#SBATCH --output=slurm.%N.%j.out
#SBATCH --export=ALL

set echo
set -x

source /opt/packages/gromacs-CPU-2018/bin/GMXRC.bash
module load gromacs/2018_cpu
module mpi/pgi_openmpi/19.10

cd $SLURM_SUBMIT_DIR

#set variable so that task placement works as expected
export  I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

#copy input files to LOCAL file storage
rsync -aP $PWD $LOCAL/

# pdb2gmx_mpi - use
# echo "4" | gmx_mpi pdb2gmx_mpi -f fff_uncap_opt_avo.pdb -o fff.gro -water spce -ignh

# define 4nm cubic box centered on a single peptide
# gmx editconf -f fff.gro -box 4 4 4 -o fff_box.gro -c

# add 24 randomly rotated peptides to that box
# gmx_mpi insert-molecules -f fff_box.gro -ci fff.gro -o fff25.gro -nmol 24 -rot xyz -seed 777

# update [molecules] directive in topol.top to include 25 fff
# optionally change residue names to improve votca cg settings readbility later

# energy minimization. this will be a short one because things look fine, but
# it's probably good practice.
mpirun -np $SLURM_NPROCS gmx_mpi grompp -f minim.mdp -c fff25.gro -p topol.top -o fff25em.tpr
mpirun -np $SLURM_NPROCS gmx_mpi mdrun -v -deffnm fff25em

# solvate then em again. some worthy graphics commented out if you're doing this
# interactively
mpirun -np $SLURM_NPROCS gmx_mpi solvate -cp fff25em.gro -cs spc216.gro -o fff25_solv.gro -p topol.top
mpirun -np $SLURM_NPROCS gmx_mpi grompp -f minim.mdp -c fff25_solv.gro -p topol.top -o fff25_solv_em.tpr
mpirun -np $SLURM_NPROCS gmx_mpi mdrun -v -deffnm fff25_solv_em
# echo "10 0" | gmx_mpi energy -f fff25_solv_em.edr -o potential.xvg

# nvt equilibration
mpirun -np $SLURM_NPROCS gmx_mpi grompp -f nvt.mdp -c fff25_solv_em.gro -r fff25_solv_em.gro -p topol.top -o nvt.tpr
mpirun -np $SLURM_NPROCS gmx_mpi mdrun -v -deffnm nvt
# echo "16 0" | gmx_mpi energy -f nvt.edr -o temperature.xvg

# npt equilibration
mpirun -np $SLURM_NPROCS gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
mpirun -np $SLURM_NPROCS gmx_mpi mdrun -v -deffnm npt
# echo "18 0" | gmx_mpi energy -f npt.edr -o pressure.xvg
# echo "24 0" | gmx_mpi energy -f npt.edr -o density.xvg

# production md
mpirun -np $SLURM_NPROCS gmx_mpi grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md_200.tpr
mpirun -np $SLURM_NPROCS gmx_mpi mdrun -v -deffnm md_200

# md analysis
# gmx_mpi trjconv -s md_0_1.tpr -f md_0_1.xtc -o md_0_1_noPBC.xtc -pbc mol -center
# gmx_mpi rms -s md_0_1.tpr -f md_0_1_noPBC.xtc -o rmsd.xvg -tu ns
# gmx_mpi rms -s em.tpr -f md_0_1_noPBC.xtc -o rmsd_xtal.xvg -tu ns
# gmx_mpi gyrate -s md_0_1.tpr -f md_0_1_noPBC.xtc -o gyrate.xvg

#Copy output files to pylon5
rsync -aP $LOCAL/ /pylon5/mr560ip/hooten/fff25_200/output

sacct --format MaxRSS,Elapsed -j $SLURM_JOBID
