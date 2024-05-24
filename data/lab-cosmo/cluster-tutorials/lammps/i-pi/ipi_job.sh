#!/bin/bash
#SBATCH --job-name=ipi
#SBATCH --get-user-env
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --exclusive

module load intel
module load intel-mkl
module load intel-mpi
module load lammps

# please check if this is your i-pi directory
IPI=$HOME/code/i-pi/bin/i-pi
LAMMPS=lmp_mpi

cd graphene
# delete ipi socket file
rm /tmp/ipi_graphene

echo ${SLURM_NTASKS}

sleep 1
${IPI} input.xml > i-pi.log &
sleep 5

srun --time=24:00:00 --hint=nomultithread --exclusive -n ${SLURM_NTASKS} ${LAMMPS} -i in.lmp > log.lammps &

wait
exit 0

