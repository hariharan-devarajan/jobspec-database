#!/bin/bash
#SBATCH --job-name=petscDMDA       # nom du job
#SBATCH --nodes=1                  # Nombre de noeud
#SBATCH --ntasks=4                 # Nombre total de processus MPI
#SBATCH --hint=nomultithread       # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=00:10:00            # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=petscDMDA%j.out   # Nom du fichier de sortie
#SBATCH --error=petscDMDA%j.out    # Nom du fichier d'erreur (ici commun avec la sortie)

# on se place dans le répertoire de soumission
cd ${SLURM_SUBMIT_DIR}

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
source ../../../petsc.sh

# echo des commandes lancées
set -x

# exécution du code
time srun ./dmda.exe -da_grid_x 1000 -da_grid_y 1000 -ksp_type cg -pc_type hypre
