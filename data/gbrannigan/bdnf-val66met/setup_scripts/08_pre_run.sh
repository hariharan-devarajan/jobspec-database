#!/bin/bash
#SBATCH -J V4           # job name
#SBATCH -o stam.out       # output and error file name (%j expands to jobID)
#SBATCH -N 1 --tasks-per-node 16
#SBATCH -p gpu         # queue (partition) -- normal, development, etc.
#SBATCH -t 1:30:00        # run time (hh:mm:ss) - 1.5 hours
##SBATCH --mail-user=ruchi15@gmail.com       
##SBATCH --mail-type=begin  # email me when the job starts
##SBATCH --mail-type=end    # email me when the job finishes

#set -x
module purge
module load intel/15.0.2
module load mvapich2/2.1
module load boost
module load cuda/7.0
module load gromacs/5.1.2
module list
export GMX_MAXBACKUP=-1
export OMP_NUM_THREADS=15


module list
( for i in {0..63} ; do cd $i ; echo 0 > pro ; ibrun -np 1  gmx grompp -f rep_"$i".mdp -t ex.cpt -c nvt_vol.gro -p topol.top -o ex.tpr  >& grompp_11.out ; cd .. ; done ) 
ibrun -np 480 mdrun_mpi_gpu -noappend -v -deffnm ex -multidir {0..63} -maxh 1 -replex 500 -nstlist 20
