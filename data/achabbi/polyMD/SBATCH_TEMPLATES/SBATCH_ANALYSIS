#!/bin/bash

#SBATCH --account=pi-depablo
#SBATCH --time=06:00:00

#SBATCH --partition=depablo-tc
#SBATCH --qos=depablo-tc-sn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10000

#SBATCH --output=LOG/NAME2_analysis.ou
#SBATCH --error=LOG/NAME2_analysis.err
#SBATCH --job-name=NAME2_analysis

######################### FILE PATHS #########################
CONF="/project2/depablo/achabbi/scripts/conformation.py"


######################### LOAD MODULES #########################
module load gromacs/2022.4+oneapi-2021
module load python/anaconda-2021.05


######################### RUN GROMACS #########################
gmx_mpi energy -f npt/npt.edr -o npt/density.xvg -b 5000 <<< density | grep "Density" | tail -1 | awk '{print $2, $3}' >> numbers.txt

gmx_mpi msd -f nvt/nvt.xtc -s nvt/nvt.tpr -o nvt/msd.xvg -sel 0 -beginfit 30000 -endfit 70000
grep 'D\[    System\]' nvt/msd.xvg | sed 's/)//g' | awk '{print $7, $9}' >> numbers.txt

gmx_mpi polystat -f nvt/nvt.xtc -s nvt/nvt.tpr -o nvt/conformation.xvg <<< 0
python ${CONF} nvt/conformation.xvg numbers.txt
