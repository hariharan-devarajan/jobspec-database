#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --threads-per-core=1
#SBATCH --distribution=block:block:block
##SBATCH --reservation=GS-20880
##SBATCH --account=courses0100
#SBATCH --job-name=fullNodeMeasures
#SBATCH --time=00:10:00
#SBATCH --mem=117Gb


#-------------------------
# Load the correct modules
#module swap gcc intel
#module load intel-mkl
module load PrgEnv-cray

#-------------------------
# Define environment variables
export OMP_PROC_BIND=SPREAD

#------------------------
#Measuring peak performance (note "optimal" size of matrix changes)
echo -e "\n\n========================================"
echo        "========================================"
export OMP_NUM_THREADS=1
echo "Measuring Peak Performance THREADS=$OMP_NUM_THREADS"
echo "Using: srun -l -u -n 1 -c $OMP_NUM_THREADS dgemm 1600 | sort -n"
srun -l -u -n 1 -c $OMP_NUM_THREADS dgemm 1600 | sort -n
#==
echo -e "\n\n========================================"
echo        "========================================"
export OMP_NUM_THREADS=1
echo "Measuring Peak Performance full chip THREADS=$OMP_NUM_THREADS"
echo "Using: srun -l -u -n 64 -c $OMP_NUM_THREADS dgemm 1600 |sort -n"
srun -l -u -n 64 -c $OMP_NUM_THREADS dgemm 1600 | sort -n

#-------------------------
#Measuring peak memory bandwith
export OMP_NUM_THREADS=1
echo -e "\n\n========================================"
echo        "========================================"
echo "Measuring Peak Mem Bandwidth THREADS=$OMP_NUM_THREADS"
echo "With srun using: srun -c $SLURM_CPUS_PER_TASK ./stream"
srun -c $SLURM_CPUS_PER_TASK ./stream
#==
echo -e "\n\n========================================"
echo        "========================================"
echo "Measuring Peak Mem Bandwidth THREADS=$OMP_NUM_THREADS"
echo "With srun using: srun -c $OMP_NUM_THREADS ./stream"
srun -c $OMP_NUM_THREADS ./stream
#==
echo -e "\n\n========================================"
echo        "========================================"
echo "Measuring Peak Mem Bandwidth THREADS=$OMP_NUM_THREADS"
echo "WithOUT srun using: ./stream"
./stream

#-------------------------
#Final steps
echo -e "\n\n========================================"
echo        "========================================"
echo "Done"
