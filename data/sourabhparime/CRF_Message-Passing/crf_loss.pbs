#!/bin/bash
#PBS -l nodes=3:ppn=20,walltime=30:00 
#PBS -N crfloss
#PBS -q edu_shared

#PBS -m abe
#PBS -M sparim2@uic.edu

#PBS -e crf_loss.err
#PBS -o crf_loss.out
#PBS -d /mnt/store3/classes/cs594/home/sparim2/project2/upload/code_PETSc

module load tools/mpich2-1.5-gcc

mpirun -machinefile $PBS_NODEFILE -np $PBS_NP ./seq_train -data ./train_petsc.txt -tdata ./test_petsc.txt -lambda 1e-3 -loss CRF -tol 1e-3
