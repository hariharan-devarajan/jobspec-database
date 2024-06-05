#!/bin/bash
#SBATCH -J V4           # job name
#SBATCH -o stam.out       # output and error file name (%j expands to jobID)
#SBATCH -n 1          # total number of mpi tasks requested
#SBATCH -p normal         # queue (partition) -- normal, development, etc.
#SBATCH -t 00:30:00        # run time (hh:mm:ss) - 1.5 hours
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
rm -rf sol.dat

for i in {0..63}
do
cd $i
ibrun -np 1 gmx pdb2gmx -h
ibrun -np 1 gmx pdb2gmx -f resid_23-113-capped.pdb -ignh -ff amber99sb-star-ildn-q -water tip4p -v -o resid_23-113-capped.gro >& pdb2gmx_1.out && \
ibrun -np 1 gmx editconf -f resid_23-113-capped.gro -o resid_23-113-capped-newbox_1nm.gro -box 11 -bt dodecahedron -c >& editconf_2.out && \
sed -i -e 's/tip4p.itp/tip4p-D.itp/g' topol.top && \
ibrun -np 1 gmx solvate -cp resid_23-113-capped-newbox_1nm.gro -cs amber99sb-star-ildn-q.ff/tip4p-D.gro -o NMR-capped-solvated.gro -p topol.top >& genbox_3.out && \


grep SOL topol.top | awk '{print $2}' >> ../sol.dat && \
cd ..
done 


avg_sol=$(echo "" | awk '{ sum += $1 } END { if (NR > 0) print sum / NR }' sol.dat)
lowest_sol=$(echo "" | sort -r sol.dat | tail -1)



#the box is resolvated with the same number of water molecules



for i in {0..63}
do
cd $i
ibrun -np 1 gmx pdb2gmx -f resid_23-113-capped.pdb -ignh -ff amber99sb-star-ildn-q -water tip4p -v -o resid_23-113-capped.gro >& pdb2gmx_1.out && \
ibrun -np 1 gmx editconf -f resid_23-113-capped.gro -o resid_23-113-capped-newbox_1nm.gro -box 11 -bt dodecahedron -c >& editconf_2.out && \
sed -i -e 's/tip4p.itp/tip4p-D.itp/g' topol.top && \
ibrun -np 1 gmx solvate -cp resid_23-113-capped-newbox_1nm.gro -cs amber99sb-star-ildn-q.ff/tip4p-D.gro -o NMR-capped-solvated.gro -p topol.top -maxsol $lowest_sol >& genbox_3.out
cd ..
done

