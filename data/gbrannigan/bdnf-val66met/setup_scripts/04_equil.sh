#!/bin/bash
#SBATCH -J Y4           # job name
#SBATCH -o stam.out       # output and error file name (%j expands to jobID)
#SBATCH -N 32 --tasks-per-node 16
#SBATCH -p normal         # queue (partition) -- normal, development, etc.
#SBATCH -t 48:00:00        # run time (hh:mm:ss) - 1.5 hours
##SBATCH --mail-user=ruchi15@gmail.com       
##SBATCH --mail-type=begin  # email me when the job starts
##SBATCH --mail-type=end    # email me when the job finishes

#set -x
module purge
module load intel/15.0.2
module load mvapich2/2.1
module load boost
#module load cuda/7.0
module load gromacs/5.1.2
module list
export GMX_MAXBACKUP=-1
export OMP_NUM_THREADS=2 # gives 4-12ns/day performance,15-6ns/day,1-7ns/day,2-4.log,8-5.log,4-6.log 

REP=$(seq 0 63)


module list
mdp=/scratch/02780/ruchi/trex-04/mdp_files

echo "13" > ../inp
for i in $REP
do
cd $i
ibrun -np 1 gmx grompp -f $mdp/em.mdp -c pre_em.gro -p topol.top -o em.tpr >& grompp_6.out
cd ..
done
ibrun -np $SLURM_NTASKS mdrun_mpi -deffnm em -multidir $REP

for i in $REP
do
cd $i
ibrun -np 1 gmx grompp -f $mdp/nvt.mdp -c em.gro -p topol.top -o nvt.tpr >& grompp_7.out
cd ..
done
ibrun -np $SLURM_NTASKS mdrun_mpi -deffnm nvt -multidir $REP

for i in $REP
do
cd $i
ibrun -np 1 gmx grompp -f $mdp/npt.mdp -c nvt.gro -p topol.top -o npt.tpr -t nvt.cpt >& grompp_8.out 
cd ..
done
ibrun -np $SLURM_NTASKS mdrun_mpi -v -noappend -deffnm npt -multidir $REP
