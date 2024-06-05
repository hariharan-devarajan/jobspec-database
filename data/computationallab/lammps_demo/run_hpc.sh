#!/usr/bin/env bash
#SBATCH -J melt
#SBATCH -o melt.stdout
#SBATCH -e melt.stderr
#SBATCH --mail-user UCID@njit.edu
#SBATCH --mail-type=ALL
#SBATCH --partition gor
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks-per-node=8

module load singularity gnu8 openmpi3

rm -rf out
mkdir out/

mpirun -np ${SLURM_NTASKS} singularity exec /opt/site/singularity-apps/lammps/20200505/lammps-20200505-centos7-python3.6.9.sif lammps -i melt.in

