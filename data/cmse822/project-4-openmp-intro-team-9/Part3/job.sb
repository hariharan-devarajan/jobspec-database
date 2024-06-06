#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########

#SBATCH --exclusive 
#SBATCH --time=00:10:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --ntasks-per-node=4                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --nodes=1          # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=8G                    # memory required per node - amount of memory (in bytes)
#SBATCH --job-name p4p1      # you can give your job a name for easier identification (same as -J)
#SBATCH --constraint="amd20"
 
########## Command Lines to Run ##########


# Specify the directory containing your executable files
cd $SLURM_SUBMIT_DIR   
mpic++ -fopenmp matmulti_omp_mpi.cpp -o matmulti_omp_mpi
N=2000
# Always in bytes
for(( size = 1; size <=4; size *= 2))
do
    for(( threads = 1; threads <= 128; threads *= 2 ))
    do
        echo "-----------------------------------------------"
        echo "Running for number of threads:$threads and nodes:$size"
        echo "-----------------------------------------------"
        mpiexec -n $size ./matmulti_omp_mpi $threads $N
    done
done


scontrol show job $SLURM_JOB_ID     ### write job information to output file
