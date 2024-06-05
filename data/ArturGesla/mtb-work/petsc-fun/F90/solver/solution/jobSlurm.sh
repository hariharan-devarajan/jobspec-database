#!/bin/bash
#SBATCH --job-name=petscKSP        # nom du job
#SBATCH --nodes=1                 # Nombre de noeud
#SBATCH --ntasks=4                 # Nombre total de processus MPI
#SBATCH --hint=nomultithread       # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=00:10:00            # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=petscKSP%j.out    # Nom du fichier de sortie
#SBATCH --error=petscKSP%j.out     # Nom du fichier d'erreur (ici commun avec la sortie)

# on se place dans le répertoire de soumission
cd ${SLURM_SUBMIT_DIR}

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
source ../../../petsc.sh

# echo des commandes lancées
set -x

# exécution du code
time srun ./solver.exe -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type cg -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type minres -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type bcgs -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type cg -pc_type jacobi -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type cg -pc_type asm -size 1000 -ksp_max_it 2000 # No convergence because non-symmetric preconditioner.
time srun ./solver.exe -ksp_type gmres -pc_type asm -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type cg -pc_type hypre -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type cg -pc_type jacobi -ksp_norm_type unpreconditioned -size 1000
echo "---------------------------------------------------------------------------------------"
time srun ./solver.exe -ksp_type cg -pc_type hypre -ksp_norm_type unpreconditioned -size 1000
