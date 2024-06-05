#!/bin/bash

#SBATCH --job-name=pvserver
#SBATCH --output=log.pvserver
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=k2-medpri,medpri
#SBATCH --time=03:00:00


module load apps/paraview/5.11.2
echo starting xvfb
#xvfb-run -s'screen 0 640x480x24' pvserver
xvfb-run echo \$DISPLAY
echo svfb finished
#pvserver --force-offscreen-rendering
#mpiexec -n 8 pvserver --force-offscreen-rendering
