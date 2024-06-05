#!/bin/bash

#SBATCH -N 1
#SBATCH -n 15
##SBATCH -q medium
#SBATCH --exclusive
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err

sleep 20

outdir=$1
tr_size=$2
working_directory=$PWD
cd $working_directory/bin
module load gcc openmpi


srun --ntasks-per-node=1 -n $SLURM_NNODES sync 
sleep 20
srun --ntasks-per-node=1 -n $SLURM_NNODES sync 
sleep 20

#-t N : transferSize size of transfer in bytes (e.g.: 8, 4k, 2m, 1g)
#-b N : blockSize contiguous bytes to write per task (e.g.: 8, 4k, 2m, 1g) (Here we use 4 Giga)
#-w   : writeFile write file
#-C   : reorderTasksConstant changes task ordering to n+1 ordering for readback
#-F   :	filePerProc file-per-process
#-o   : testFile full name for test
echo "srun ./ior -t${tr_size}k -b4g -w  -C -F  -o $working_directory/$outdir/log/$SLURM_NTASKS.bin > $working_directory/$outdir/$SLURM_NTASKS.txt"
srun ./ior -t${tr_size}k -b4g -w  -C -F  -o $working_directory/$outdir/log/$SLURM_NTASKS.bin > $working_directory/$outdir/$SLURM_NTASKS.txt
