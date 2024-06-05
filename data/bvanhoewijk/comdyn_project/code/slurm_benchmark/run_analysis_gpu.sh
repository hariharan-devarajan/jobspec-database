#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --exclusive
 
module load 2022
module load GROMACS/2021.6-foss-2022a-CUDA-11.7.0

THREADS=18
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Minimization
setenv GMX_MAXCONSTRWARN -1
# step6.0 - soft-core minimization
# If you encountered "There are 1 perturbed non-bonded pair interaction ......" error message,
# please modify rvdw and rcoulomb values from 1.1 to 2.0 in the step6.0_minimization.mdp file
srun gmx grompp -f step7_production.mdp -o step7_production.tpr -c step6.6_equilibration.gro -p system.top -n index.ndx
srun gmx mdrun -deffnm step7_production -ntomp ${THREADS} -pin on -pinstride 1 -g default.log -ntmpi 1

srun gmx grompp -f step7_production.mdp -o step7_production.tpr -c step6.6_equilibration.gro -p system.top -n index.ndx
srun gmx mdrun -deffnm step7_production -ntomp ${THREADS} -pin on -pinstride 1 -g manual-nb-bonded.log -nb gpu -bonded gpu -ntmpi 1

srun gmx grompp -f step7_production.mdp -o step7_production.tpr -c step6.6_equilibration.gro -p system.top -n index.ndx
srun gmx mdrun -deffnm step7_production -ntomp ${THREADS} -pin on -pinstride 1 -g manual-nb.log -nb gpu -bonded cpu -ntmpi 1

echo Done
rm -f *cpt *edr *trr *tng *gro \#*



