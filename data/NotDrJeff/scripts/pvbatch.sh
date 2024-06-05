#!/bin/bash

#SBATCH --job-name=pvbatch
#SBATCH --output=log.pvbatch
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --partition=k2-medpri,medpri
#SBATCH --time=05:00:00
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread

if [ -f $1 ]; then
    ~/OpenFOAM/ParaView-5.11.1/bin/mpiexec -n 16 ~/OpenFOAM/ParaView-5.11.1/bin/pvbatch "$@" 2>&1
fi
