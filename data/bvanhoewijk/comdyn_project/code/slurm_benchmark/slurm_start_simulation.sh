#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --partition=genoa
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH -a 1,2

module load 2022
module load GROMACS/2021.6-foss-2022a

# Check if directories exist:
for rep in 1 2
do
mkdir -p rep${rep}
done

rep=${SLURM_ARRAY_TASK_ID}

# Production run
setenv GMX_MAXCONSTRWARN -1
export OMP_NUM_THREADS=1

# If TPR is defined:
if [ -f "rep${rep}/step7_production_rep${rep}.tpr" ]; then
    echo "TPR is defined. Skipping gmx grompp."
else
    # Create TPR:
    srun gmx grompp -f step7_production.mdp -o rep${rep}/step7_production_rep${rep}.tpr -c step6.6_equilibration.gro -p system.top -n index.ndx -po rep${rep}/mdout.mdp
fi

# If checkpoint:
if [ -f "rep${rep}/step7_production_rep${rep}.cpt" ]; then
    echo "Using checkpoint"
    mpirun gmx_mpi mdrun -cpi rep${rep}/step7_production_rep${rep}.cpt -deffnm rep${rep}/step7_production_rep${rep} -append -g rep${rep}_md.log
else
    # If no checkpoint:
    mpirun gmx_mpi mdrun -deffnm rep${rep}/step7_production_rep${rep} -cpt 1 -pin on -g rep${rep}_md.log
fi
