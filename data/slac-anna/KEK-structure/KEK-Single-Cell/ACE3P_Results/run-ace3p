#!/bin/bash -l

#SBATCH -o job.%j.out
#SBATCH -e job.%j.err

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -A m1779
#SBATCH -C haswell

cd $SLURM_SUBMIT_DIR

module -s swap PrgEnv-intel PrgEnv-gnu
module load cray-petsc
module load cray-parallel-netcdf
module load cray-hdf5-parallel
module load gsl/2.5
module load cray-tpsl
module load arpack-ng/3.7.0-gcc
module load cray-trilinos

srun -n 1 --cpu_bind=cores /global/cfs/cdirs/ace3p/cori/acdtool meshconvert SingleCell.gen
srun -n 32 --cpu_bind=cores /global/cfs/cdirs/ace3p/cori/omega3p singlecell.omega3p
srun -n 32 --cpu_bind=cores /global/cfs/cdirs/ace3p/cori/track3p singlecell.track3p

