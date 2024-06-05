#!/bin/bash
#SBATCH -J G1_vesicle
#SBATCH -o G1.out
#SBATCH -e G1.err
#SBATCH -N 8
#SBATCH -n 512
#SBATCH -p development
#SBATCH -t 02:00:00
#SBATCH -A YYYYY
#SBATCH --mail-user=YYYYY
#SBATCH --mail-type=all

tar xvf package.tar.gz
module load gromacs

STAMPEDE2PROCS=512

MPI_GMX_EXEC=/opt/apps/intel18/impi18_0/gromacs/2018.3/bin/gmx_knl
MPI_MDRUN_EXEC=/opt/apps/intel18/impi18_0/gromacs/2018.3/bin/mdrun_mpi_knl

extra_pins="-pin on -resethway"

start=26
skip=2
###################### NPT 323K 60 ns 5 fs #####################################

i=$((start+skip-2))

echo $i
GRO=$i-input.gro
tpr=$((++i))-input
minim=npt_5fs_long.mdp
top=system.top
out=$((++i))-input

#Replace below "em" by your input file name

$MPI_GMX_EXEC grompp -f package/$minim  -c $GRO -p $top -o $tpr

ibrun -n $STAMPEDE2PROCS $MPI_MDRUN_EXEC -v -deffnm $out -s $tpr $extra_pins


###################### NPT 323K 400 ns 10 fs #####################################




