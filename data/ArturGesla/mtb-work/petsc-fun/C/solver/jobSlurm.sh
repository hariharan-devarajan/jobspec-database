#!/bin/bash
#SBATCH --job-name=petscKSP        # nom du job
#SBATCH --nodes=1                  # Nombre de noeud
#SBATCH --ntasks=8                 # Nombre total de processus MPI
#SBATCH --hint=nomultithread       # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=00:10:00            # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=petscKSP%j.out    # Nom du fichier de sortie
#SBATCH --error=petscKSP%j.out     # Nom du fichier d'erreur (ici commun avec la sortie)


# on se place dans le répertoire de soumission
cd ${SLURM_SUBMIT_DIR}

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
source ../../petsc.sh

# echo des commandes lancées
#set -x

# exécution du code
n=1000;
#time srun ./solver.exe -ksp_monitor -size $n 
#echo "it timing";
time srun ./solver.exe -size $n -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -log_view
echo "dir timing";

