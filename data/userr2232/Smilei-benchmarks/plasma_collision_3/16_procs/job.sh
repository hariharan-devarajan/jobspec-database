#!/bin/bash
#SBATCH --job-name smilei
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --mem-per-cpu=1gb

srun --mpi=pmix_v2 /home/reynaldo.rojas/smilei/Smilei/Smilei-benchmarks/plasma_collision_3/16_procs/smilei /home/reynaldo.rojas/smilei/Smilei/Smilei-benchmarks/plasma_collision_3/16_procs/plasma_collision.py
