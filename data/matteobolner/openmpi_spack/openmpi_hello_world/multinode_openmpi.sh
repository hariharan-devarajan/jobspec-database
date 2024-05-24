#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --partition=skl_usr_prod

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1


#SBATCH --out=myJob.out 
#SBATCH --err=myJob.err 
#SBATCH --account=ExaF_prod_0

#module load gnu/7.3.0
#module load openmpi/3.0.0--gnu--7.3.0

#mpicc hello.c -o hello
#/marconi_work/ExaF_prod_0/spack/opt/spack/linux-centos7-broadwell/gcc-7.3.0/openmpi-3.1.4-ymdp6w7f2dkp7miqsnpuygnk72hzi4n4/bin/mpicc hello.c -o hello

#srun ./hello
#/marconi_work/ExaF_prod_0/spack/opt/spack/linux-centos7-broadwell/gcc-7.3.0/openmpi-3.1.4-ymdp6w7f2dkp7miqsnpuygnk72hzi4n4/bin/mpirun -display-allocation ./hello 

module use /marconi_work/ExaF_prod_0/spack/opt/spack/linux-centos7-broadwell/gcc-7.3.0/openmpi-4.0.1-tsuqdoly7rjs3vy6dk5pugjj4so3cu26

module load openmpi-4.0.1-gcc-7.3.0-tsuqdol

srun -display-allocation -N $SLURM_NTASKS hello
