#!/bin/sh
#PBS -l nodes=1:ppn=1
### Merge stderr with stdout
#PBS -j oe
### Queue name
#PBS -q default
###Job name
#PBS -N CNT
### Declare job-non-rerunable
#PBS -r n
#PBS -V
# This job's working directory
echo Job ID: $PBS_JOBID
echo Working directory is $PBS_O_WORKDIR cd $PBS_O_WORKDIR echo Running on host `hostname` echo Time is `date` echo Directory is `pwd` echo This job runs on the following processors:
echo `cat $PBS_NODEFILE`


$ cat > testfor
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64
do
	RUNPATH=/home/jason/phonopy/cnt/POSCAR$i
	EXEPATH=/home/jason/lammps/lammps-10Oct10/src
	cd $RUNPATH
	/opt/open-mpi/tcp-gnu41/bin/mpirun -np 1 $EXEPATH/lmp_openmpi < $RUNPATH/in.force
done



