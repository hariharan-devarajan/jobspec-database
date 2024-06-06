
#!/bin/bash
#BSUB -nnodes 2  
#BSUB -W 120 
#BSUB -qpdebug


module load cuda/11.1.1  gcc/8.3.1 

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(spack location -i gsl)/lib mpirun -np 4 ./sdl_allreduce 4194304 >output_4194304_4h
