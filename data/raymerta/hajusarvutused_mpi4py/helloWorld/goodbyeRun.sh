
#!/bin/bash
#SBATCH -J Goodbye
#SBATCH -N 2
#SBATCH --ntasks-per-node=2

module purge
module load openmpi-1.7.3
module load python-2.7.3
export MPI4PYDIR=paralleelarvutused
export PYTHONPATH=$HOME/$MPI4PYDIR/install/lib/python

mpirun python helloworld.py