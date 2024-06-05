#!/bin/bash
#SBATCH -J V4           # job name
#SBATCH -o stam4.out       # output and error file name (%j expands to jobID)
#SBATCH -N 1 --tasks-per-node 16
#SBATCH -p normal         # queue (partition) -- normal, development, etc.
#SBATCH -t 1:00:00        # run time (hh:mm:ss) - 1.5 hours
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




module list
mdp=/scratch/02780/ruchi/trex-03/mdp_files #mdp_files folder is provided

echo "13" > inp
for i in {0..63}
do
cd $i
ibrun -np 1 gmx grompp -f $mdp/em.mdp -c NMR-capped-solvated.gro -p topol.top -o ions.tpr >& grompp_4.out && \
ibrun -np 1 gmx genion -s ions.tpr -o pre_em.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15 < ../inp >& genion_5.out && \
cd ..
done



