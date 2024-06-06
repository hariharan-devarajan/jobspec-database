#!/bin/bash
# Begin LSF Directives
#BSUB -P CHM155_003
#BSUB -W 0:20
#BSUB -nnodes 64
#BSUB -alloc_flags "gpumps gpudefault"
#BSUB -J aorta_NoCuAw_v1_27a
#BSUB -o aorta_NoCuAw_v1_27a.%J
#BSUB -e aorta_NoCuAw_v1_27a.%J

date

module load gcc cuda cmake/3.14.2

EXEC_PATH=/gpfs/alpine/proj-shared/chm155/izacharo/hemeLB/GNU/v1_27a/src_v1_27/build
EXEC=hemepure_gpu_VelPresBCs_NoCuAwMPI_F1e4_SwapPoint

rm -r results

jsrun --nrs 384 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 6  --launch_distribution packed -\
-bin\
d packed:1 $EXEC_PATH/$EXEC -in input.xml -out results

