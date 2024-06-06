#!/bin/bash -l
#SBATCH --account=pawsey0110
#SBATCH --job-name=premd_100nM_Na_Aqvist
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --export=All

#======START=====
module load slurm
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes"
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"
echo "Node list:"
sacct --format=JobID,NodeList%100 -j $SLURM_JOB_ID

module load gromacs/2023

#These steps help slurm distribute jobs appropriately over cpus
export OMP_NUM_THREADS=1
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

#Submit script with sbatch. For more info see:
#https://support.pawsey.org.au/documentation/display/US/Job+Scheduling#JobScheduling-Batchjobs

#When starting a new procedure, it's always safest to run grompp steps interactively
#before submitting to the queue, in case it fails and you need to tweak something. I
#always include it here anyway to keep a record of it, but comment it out.

#gmx_mpi_d grompp -f premd1.mdp -c en_min2.gro -r en_min2.gro -p L21hybrid_bilayer_topol.top -n index.ndx -o premd1.tpr
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK -m block:block:block gmx_mpi_d mdrun -dlb yes -deffnm premd1

srun -N 1 -n 1 -c 1 gmx_mpi_d grompp -f premd2.mdp -c premd1.gro -p L21hybrid_bilayer_topol.top -n index.ndx -o premd2.tpr
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK -m block:block:block gmx_mpi_d mdrun -dlb yes -deffnm premd2
