#!/bin/bash
#SBATCH -J V4           # job name
#SBATCH -o stam.out       # output and error file name (%j expands to jobID)
#SBATCH -n 64          # total number of mpi tasks requested
#SBATCH -p normal         # queue (partition) -- normal, development, etc.
#SBATCH -t 5:30:00        # run time (hh:mm:ss) - 1.5 hours
##SBATCH --mail-user=ruchi15@gmail.com       
##SBATCH --mail-type=begin  # email me when the job starts
##SBATCH --mail-type=end    # email me when the job finishes

#set -x
module purge
module load intel/15.0.2
module load mvapich2/2.1
module load boost
module load gromacs/5.1.2
module list
export GMX_MAXBACKUP=-1



module list
mdp=/scratch/02780/ruchi/trex-03/mdp_files

#run the equilibration for each replica

echo "13" > ../inp
ibrun -np 1  gmx grompp -f $mdp/em.mdp -c eq_vol.gro -p topol.top -o em_vol.tpr >& grompp_9.out && \
ibrun -np 64 mdrun_mpi -deffnm em_vol  && \
ibrun -np  1 gmx grompp -f $mdp/nvt.mdp -c em_vol.gro -p topol.top -o nvt_vol.tpr >& grompp_10.out && \
ibrun -np 64 mdrun_mpi -deffnm nvt_vol 
