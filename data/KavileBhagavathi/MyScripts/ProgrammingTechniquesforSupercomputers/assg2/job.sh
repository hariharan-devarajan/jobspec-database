#!/bin/bash -l
#SBATCH --nodes=1 --time=01:00:00
#SBATCH --job-name=streamtriad_daxpy_benchmarking
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module purge
module load intel
icx -O3 -xHost -fno-alias assignment2_VectorTriad.c time.c -o vectortriad 
icx -O3 -xHost -fno-alias assignment2_DAXPY.c time.c -o daxpy
icx -O3 -xHost -fno-alias scan.c time.c
echo "Running benchmarks"
#for ((i=0;i<5;i++)) do
    #srun --cpu-freq=2400000-2400000 ./a.out
#done 
srun --cpu-freq=2200000-2200000 ./vectortriad
srun --cpu-freq=2200000-2200000 ./daxpy

