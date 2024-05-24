
## Running lammpsGPU 
```
#!/bin/bash
#SBATCH --job-name=XXX
#SBATCH --account=XXXXX
#SBATCH --nodes=1           
#SBATCH --ntasks-per-node=16     
#SBATCH --gpus-per-node=3     
#SBATCH --time=148:00:00          

module load rcac
module load ngc
 
srun --mpi=pmi2 \
singularity run --nv -B ${PWD}:/host_pwd  lammpsGPU2022.sif \
lmp -k on g 3 -sf kk -pk kokkos cuda/aware on -in stay1000K.in >stay.rec

```