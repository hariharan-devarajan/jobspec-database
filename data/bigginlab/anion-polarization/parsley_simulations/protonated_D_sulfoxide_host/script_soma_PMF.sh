#!/bin/bash
#!/bin/bash
#SBATCH --nodes=1
## The use of ntasks-per-socket instead of ntasks-per-node is *very* important
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=144:00:00
#SBATCH -J D_Cl_water_parsley
#SBATCH -p gpu-biggin
#SBATCH --array=0-23:1%2

###SLURM_ARRAY_TASK_ID=20
###SLURM_CPUS_PER_TASK=1
echo ${SLURM_ARRAY_TASK_ID}

module add apps/gromacs/gcc-10.3/2021.4
export OMP_NUM_THREADS=${​​​​SLURM_CPUS_PER_TASK}
echo ${SLURM_CPUS_PER_TASK}

mkdir EQUIL
mkdir PROD
mkdir unrestr

fname=D_sulfoxide_system #ligand_solv_ions
folder=EQUIL

for i in $((${SLURM_ARRAY_TASK_ID}))  # {-49..-46} $((-${SLURM_ARRAY_TASK_ID}))
do
   echo $i
   #gmx grompp -f MDP/MIN.mdp -c $fname.gro -r $fname.gro -p topol.top -n index.ndx -o $folder/min.tpr
   cd unrestr
   #gmx mdrun -v -deffnm min
   cd ..
   ### NVT k=500 ###
   gmx grompp -f MDP/NVT_$i.mdp -c unrestr/npt_0.gro -r $fname.gro -p topol.top -n index.ndx -o $folder/nvt_$i.tpr
   cd $folder
   gmx mdrun -v -deffnm nvt_$i -ntomp ${SLURM_CPUS_PER_TASK} -update gpu
   cd ..
   ### 100ps NPT k=1000 ###
   gmx grompp -f MDP/NPT_$i.mdp -c $folder/nvt_$i.gro -r $fname.gro -t $folder/nvt_$i.cpt -p topol.top -n index.ndx -o $folder/npt_$i.tpr
   cd  $folder
   gmx mdrun -v -deffnm npt_$i -ntomp ${SLURM_CPUS_PER_TASK} -update gpu
   cd ..
   ### 10 ns k=500 ###
   gmx grompp -f MDP/PROD_$i.mdp -c $folder/npt_$i.gro -r $fname.gro -t $folder/npt_$i.cpt -p topol.top -n index.ndx -o PROD/prod_$i.tpr
   cd PROD
   gmx mdrun -v -deffnm prod_$i -ntomp ${SLURM_CPUS_PER_TASK} -update gpu
   cd ..
done
