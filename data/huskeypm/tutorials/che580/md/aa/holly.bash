#!/bin/sh
#SBATCH -p limited # partition (queue) for CHE580 
# set max wallclock time 
#SBATCH --time=30:00:00
#SBATCH --tasks-per-node=2
# module load NAMD/2.9 
module load NAMD

# Run jobs
NP=$SLURM_NTASKS
echo "Running with $NP tasks" 

# export EXEC="mpirun -np $NP /share/apps/NAMD/2.9/Linux-x86_64-ibverbs/namd2"
export EXEC="charmrun ++p $NP ++mpiexec /share/apps/NAMD/2.9/Linux-x86_64-ibverbs/namd2"
            #charmrun ++p   2 ++mpiexec /share/apps/NAMD/2.9/Linux-x86_64-ibverbs/namd2 production_iter30.conf
# For now, until new version is recompiled 
# export EXEC="/share/apps/NAMD/2.9/Linux-x86_64-ibverbs/namd2"

${EXEC} run.conf >& run.out 

# module load amber
# cpptraj lys_QwikMD.psf < inp.cpptraj
