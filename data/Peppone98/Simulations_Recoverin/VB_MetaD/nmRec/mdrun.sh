#!/bin/bash
#SBATCH --job-name VB_nmRec
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_Meta-Rec
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal



module load profile/lifesc
module unload fftw
module load gromacs/2021.7--openmpi--4.1.4--gcc--11.3.0-cuda-11.8

export OMP_NUM_THREADS=8

echo -n "Starting Script at: "
date

wait
gmx_mpi mdrun -s md_Meta.tpr -plumed VB_MetaD.dat -ntomp 8 -v -nb gpu -pme auto -pin off

wait
echo ""
echo "Finished first benchmark for VB meta!!"
echo ""

echo "done at"
date
