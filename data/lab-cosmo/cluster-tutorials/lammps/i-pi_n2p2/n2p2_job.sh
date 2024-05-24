#!/bin/bash
#SBATCH --job-name=n2p2
#SBATCH --get-user-env
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH --time=24:00:00

source $HOME/venv/base/bin/activate

module load intel
module load intel-mkl
module load intel-mpi
module load lammps

# please check if this is your i-pi directory
IPI=$HOME/code/i-pi/bin/i-pi
LAMMPS=lmp_mpi

cd scpimd1hvt4
# remove ipi socket file
rm /tmp/ipi_scpimd1hvt4

sleep 1
${IPI} input.xml > i-pi.log &
sleep 5
srun --time=24:00:00 --hint=nomultithread --exclusive -n ${SLURM_NTASKS} ${LAMMPS} -i in.lmp > log.lammps &

wait
exit 0
