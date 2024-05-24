#!/bin/sh
#
#SBATCH --job-name=CNN_Burn
#SBATCH --account=43299_sp0039
#
#SBATCH --partition=Nvidia
#SBATCH --cpus-per-task=40             ## number or regular CPU cores, you can use up to 20 per GPU
#SBATCH --mem=180000
#SBATCH --gres=gpu:tesla:2    ## GPU type and number, alternates: gpu:geforce:2,  gpu:tesla:1, gpu:tesla:2
#SBATCH --time=06:00:00

module load tensorflow-gpu


# paths
RUNDIR=/storage02/43299_sp0039/burn_multi/



OUTDIR=/storage02/43299_sp0039/burn_multi/output


# commands
python ${RUNDIR}/GPU.py

python ${RUNDIR}/Model.py $RUNDIR
python ${RUNDIR}/Evaluation.py $RUNDIR $OUTDIR
