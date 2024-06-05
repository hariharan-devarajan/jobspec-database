#!/bin/bash
#
#SBATCH --job-name=jobname
#SBATCH --output=jobname_%j_stdout.txt
#SBATCH --error=jobname_%j_stderr.txt
#SBATCH -p normal
#SBATCH -n 1
#SBATCH --tasks-per-node=24
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=12
#SBATCH --time=30:00:00
#SBATCH --exclusive

module load ORCA/5.0.2-gompi-2021b
module load OpenMPI/4.1.1-GCC-9.3.0-UCX-1.10.0-libfabric-1.12.1-PMIx-3.2.3

/opt/oscer/software/ORCA/5.0.2-gompi-2021b/bin/orca 2Si2Ph.inp > 2Si2Ph.out
