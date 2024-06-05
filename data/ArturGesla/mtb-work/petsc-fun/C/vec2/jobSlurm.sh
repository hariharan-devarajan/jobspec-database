#!/bin/bash
#SBATCH --job-name=petscVec        # nom du job
#SBATCH --nodes=1                  # Nombre de noeud
#SBATCH --ntasks=4                 # Nombre total de processus MPI
#SBATCH --hint=nomultithread       # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=00:10:00            # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=petscVec%j.out    # Nom du fichier de sortie
#SBATCH --error=petscVec%j.out     # Nom du fichier d'erreur (ici commun avec la sortie)

# on se place dans le répertoire de soumission
cd ${SLURM_SUBMIT_DIR}

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
source ../../petsc.sh

# echo des commandes lancées
set -x

# exécution du code
echo "-------------- Run of Vec2a ----------------"
time srun ./vec2a.exe
echo "-------------- Run of Vec2b ----------------"
time srun ./vec2b.exe
echo "-------------- Run of Vec2c ----------------"
time srun ./vec2c.exe
