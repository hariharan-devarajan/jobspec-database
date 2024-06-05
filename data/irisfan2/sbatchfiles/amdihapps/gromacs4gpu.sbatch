#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4
#SBATCH -o %x-%N-%j.err
#SBATCH -e %x-%N-%j.out

source /etc/profile.d/modules.sh
module load rocm/5.2.3

tmp=/tmp/$USER/gpu4-$$
mkdir -p $tmp

#===== ADH-DODEC Benchmark, Threaded-MPI
mkdir -p /tmp/$USER/gpu4-$$/adh_dodec
singularity run /shared/apps/bin/gromacs.sif tar -xvf /benchmarks/adh_dodec/adh_dodec.tar.gz -C /tmp/$USER/gpu4-$$/adh_dodec 1>/dev/null
singularity run /shared/apps/bin/gromacs.sif gmx mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 4 -ntomp 16 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -nstlist 200 -gpu_id 0123 -s /tmp/$USER/gpu4-$$/adh_dodec/topol.tpr -g /tmp/$USER/gpu4-$$/adh_dodec/md.log -e /tmp/$USER/gpu4-$$/adh_dodec/ener.eder -cpo /tmp/$USER/gpu4-$$/adh_dodec/state.cpt

#===== CELLULOSE_NVE Benchmark, Threaded-MPI
mkdir -p /tmp/$USER/gpu4-$$/cellulose_nve
singularity run /shared/apps/bin/gromacs.sif tar -xvf /benchmarks/cellulose_nve/cellulose_nve.tar.gz -C /tmp/$USER/gpu4-$$/cellulose_nve 1>/dev/null
singularity run /shared/apps/bin/gromacs.sif gmx mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 4 -ntomp 16 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -nstlist 200 -gpu_id 0123 -s /tmp/$USER/gpu4-$$/cellulose_nve/topol.tpr -g /tmp/$USER/gpu4-$$/cellulose_nve/md.log -e /tmp/$USER/gpu4-$$/cellulose_nve/ener.eder -cpo /tmp/$USER/gpu4-$$/cellulose_nve/state.cpt

# STMV Benchmark, Threaded-MPI
mkdir -p /tmp/$USER/gpu4-$$/stmv
singularity run /shared/apps/bin/gromacs.sif tar -xvf /benchmarks/stmv/stmv.tar.gz -C /tmp/$USER/gpu4-$$/stmv 1>/dev/null
singularity run /shared/apps/bin/gromacs.sif gmx mdrun -pin on -nsteps 100000 -resetstep 90000 -ntmpi 8 -ntomp 8 -noconfout -nb gpu -bonded gpu -pme gpu -npme 1 -v -nstlist 400 -gpu_id 0123 -s /tmp/$USER/gpu4-$$/stmv/topol.tpr -g /tmp/$USER/gpu4-$$/stmv/md.log -e /tmp/$USER/gpu4-$$/stmv/ener.eder -cpo /tmp/$USER/gpu4-$$/stmv/state.cpt

cp -r /tmp/$USER/gpu4-$$   $PWD/gpu4-$$-$SLURM_JOB_NODELIST-$SLURM_JOB_ID

# Do the clean-up
/bin/rm -rf $tmp
