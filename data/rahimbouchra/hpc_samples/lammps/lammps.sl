#!/bin/bash
#SBATCH -J LMP
#SBATCH --ntasks=16
#SBATCH --ntasks-per-core=1
#SBATCH --partition=shortq
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err


module load LAMMPS/7Aug2019-foss-2019b-Python-3.7.4-kokkos


export WORK_DIR=/data/$USER/LMP${SLURM_JOB_ID}
export INPUT_DIR=$PWD/myInput
mkdir -p $WORK_DIR
cp -R $INPUT_DIR/* $WORK_DIR
cd $WORK_DIR


echo "Running Lammps with  $SLURM_NTASKS at : $WORK_DIR"

mpirun -np $SLURM_NTASKS lmp -in indent.in.min

echo "slurm  JOB Done"
