#!/bin/bash
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=IscrC_BARTWIST
#SBATCH --partition=m100_usr_prod
#SBATCH --mail-type=END
#SBATCH --mail-user=pbarletta@gmail.com
#SBATCH --job-name nt_npt
#SBATCH -o salida_npt
#SBATCH -e error_npt
#SBATCH --depend=afterany:9749158

module load profile/lifesc
module load autoload gromacs/2022

export OMP_NUM_THREADS=8
export OMP_PLACES=threads

#set up for GPUdirect
export GMX_GPU_DD_COMMS=true
export GMX_GPU_PME_PP_COMMS=true
export GMX_FORCE_UPDATE_DEFAULT_GPU=true

num_mpi='1'

cd $SLURM_SUBMIT_DIR

gpu_idx=0
thread_idx=8
mol='d11'
mdp="/m100_work/AIRC_Fortun21/barletta/twist/run/mdp/npt_250ns.mdp"

# gmx grompp -f $mdp -c start.gro -p p2g.top -o npt_${mol}.tpr -maxwarn 5
# gmx_thread_mpi mdrun -ntomp $OMP_NUM_THREADS -ntmpi $num_mpi -nb gpu -pme gpu -bonded gpu -deffnm npt_${mol} -pin on -pinoffset ${thread_idx} -gpu_id ${gpu_idx} -v

gmx_thread_mpi mdrun -ntomp $OMP_NUM_THREADS -ntmpi $num_mpi -nb gpu -pme gpu -bonded gpu -deffnm npt_${mol} -pin on -pinoffset ${thread_idx} -gpu_id ${gpu_idx} -cpi npt_${mol}
