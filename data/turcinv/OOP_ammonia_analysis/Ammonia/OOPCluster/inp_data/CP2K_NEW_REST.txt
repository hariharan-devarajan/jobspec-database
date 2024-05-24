#!/bin/bash

#SBATCH -p cpu,scpu,bfill
#SBATCH --mem=80Gb
#SBATCH --job-name="ch102"
#SBATCH --qos=backfill
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=36
#SBATCH --time=4:00:00
#SBATCH --exclusive

# # #SBATCH --mem=0
# ## #SBATCH --constraint=bigmem
#### #SBATCH --export=NONE

# settings
input=cluster_hybrid.inp
log=cluster_hybrid.log

# export SLURM_MPI_TYPE=pmix_v3

source /opt/uochb/soft/spack/latest/share/spack/setup-env.sh #openmpi 3.1.6
spack env activate cp2k71

cp2k=cp2k.psmp

# print some information
echo ' started at:' `date`
echo '   hostname:' `hostname`
echo " "

srun --mpi=pmix $cp2k $input >> $log

# report finish time
echo 'finished at:' `date`

# restart
dt="$(date '+%d/%m/%Y')"
finish="07/05/2022"
echo $dt

#if [ "$dt" != "$finish" ]; then
#   sbatch -p cpu,scpu,bfill cp2k_new_rest.sh
#fi
