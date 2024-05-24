#!/bin/bash
#SBATCH --job-name=build_dataset_resunet-a         # Nombre del trabajo
#SBATCH --ntasks=7                        # Correr 7 tareas
#SBATCH --time=1-00:05:00                 # Timpo limite d-hrs:min:sec
#SBATCH --output=slurm_output/array_%A-%a.log  # Output (%A se reemplaza por el ID del trabajo maestro, %a se reemplaza por el indice del arreglo)
#SBATCH --error=slurm_output/array_%A-%a.err         # Output de errores (opcional)
#SBATCH -m cyclic:cyclic
#SBATCH --array=1-7                  # 100 procesos, 10 simultáneos
#SBATCH --ntasks-per-node=4

list=('' 'AT' 'ES' 'FR' 'LU' 'NL' 'SE' 'SI')

# echo ${list[SLURM_ARRAY_TASK_ID]}
# Activar ambiente de conda
# conda activate resunet-a
python /home/chocobo/Cenia-ODEPA/ResUnet-a_original/maskimg.py --country ${list[SLURM_ARRAY_TASK_ID]}