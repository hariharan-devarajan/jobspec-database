#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=160
#SBATCH --partition=genoa
#SBATCH --time=8:00:00
#SBATCH -a 1,2,3,4,5
#module load 2023
#module load GROMACS/2023.3-foss-2023a 

# Check if directories exist:
for rep in 1
do
mkdir -p rep${rep}
done

rep=${SLURM_ARRAY_TASK_ID}

############################################################################## Production run

rep=1

# If TPR is defined:
if [ -f "rep${rep}/step7_production_rep${rep}.tpr" ]; then
    echo "####################### TPR is defined. Skipping gmx grompp."
else
    # Create TPR:
    echo "####################### TPR not defined. Creating via grompp"
    gmx grompp -f step7_production.mdp -o rep${rep}/step7_production_rep${rep}.tpr -c step6.6_equilibration.gro -p system.top -n index.ndx -po rep${rep}/mdout.mdp
fi

# If checkpoint:
if [ -f "rep${rep}/step7_production_rep${rep}.cpt" ]; then
    echo "####################### Using checkpoint"
    gmx mdrun -cpi rep${rep}/step7_production_rep${rep}.cpt -deffnm rep${rep}/step7_production_rep${rep} -append -g rep${rep}/rep${rep}_md.log
else
    # If no checkpoint:
    echo "####################### No checkpoint. Starting fresh run"
    gmx mdrun -deffnm rep${rep}/step7_production_rep${rep} -cpt 1 -pin on -g rep${rep}/rep${rep}_md.log -maxh 8
fi
