#!/bin/bash
#SBATCH --output=EPYC_7H12.out # Output file (default: slurm-%j.out)
#SBATCH --error=EPYC_7H12.err # Error file  (default: slurm-%j.out)
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=1             # Number of CPUs per task
#SBATCH --mem-per-cpu=2048            # Memory per CPU
#SBATCH --time=00:01:00               # Wall clock time limit
#SBATCH --constraint=EPYC_7H12


module load gcc # load a compiler (gcc/11.4.0)
#32MB/8bytes * 4 = 16M
gcc -O3 -march=native -DSTREAM_TYPE=double -DSTREAM_ARRAY_SIZE=20000000 \
-DNTIMES=20 stream.c -o stream_c.exe # compile
./stream_c.exe