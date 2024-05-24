#!/bin/bash
#SBATCH --account=p30157
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=2:00:00
#SBATCH --job-name=test_mpi
#SBATCH --output=outlog
#SBATCH --error=errlog

source /home/sas4990/packages/spack/share/spack/setup-env.sh
module purge all

BASE_DIR=$(pwd)
rm -rf questbeta
mkdir questbeta

for i in sfwpwj3
do

    cd $BASE_DIR

    spack load --dependencies /$i

    export OMP_NUM_THREADS=1

    for run in {1..5}
    do
	    srun --mpi=pmi2 gmx_mpi mdrun -s lignocellulose-rf.tpr -maxh 0.5 -resethway -noconfout -nsteps 10000 -g </dev/null &> log_$run
    done

done 

