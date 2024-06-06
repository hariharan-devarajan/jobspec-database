#!/bin/bash
#SBATCH --job-name=lackey
#SBATCH -D .
#SBATCH --output=lackey4_%j.out
#SBATCH --error=lackey4_%j.err
#SBATCH --nodes=4
#SBATCH --time=17:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
# Uncomment the following line to trace MPI+OmpSs programs
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libnanosmpitrace.so

# Uncomment the following line to trace MPI+OpenMP (GNU) programs
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitrace.so

#export LD_PRELOAD="$LD_PRELOAD:$OMPSSEE_LD_PRELOAD"

#export EXTRAE_CONFIG_FILE=extrae.xml
#export NX_INSTRUMENTATION=extrae
x=$1

#for i in {1..10}; 
#do 
export OMP_NUM_THREADS=48
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 1 /usr/bin/time -v ./nest $x &>> n4_1.txt
#mpi2prv -f TRACE.mpits -o lackey1$i.prv
#done 


#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#for i in {1..10};
#do
export OMP_NUM_THREADS=24
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 2 /usr/bin/time -v  ./nest $x &>> n4_2.txt
#mpi2prv -f TRACE.mpits -o lackey2$i.prv
#done

#SBATCH --ntasks=3
#SBATCH --cpus-per-task=16
#for i in {1..10};
#do
export OMP_NUM_THREADS=16
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 3 /usr/bin/time -v ./nest $x &>> n4_3.txt
#mpi2prv -f TRACE.mpits -o lackey3$i.prv
#done

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#for i in {1..10};
#do
export OMP_NUM_THREADS=12
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 4 /usr/bin/time -v  ./nest $x  &>> n4_4.txt
#mpi2prv -f TRACE.mpits -o lackey4$i.prv
#done

#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#for i in {1..10};
#do
export OMP_NUM_THREADS=8
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 6 /usr/bin/time -v ./nest $x  &>> n4_6.txt
#mpi2prv -f TRACE.mpits -o lackey6$i.prv
#done

#SBATCH --ntasks=8
#SBATCH --cpus-per-task=6
#for i in {1..10};
#do
export OMP_NUM_THREADS=6
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 8 /usr/bin/time -v ./nest $x &>> n4_8.txt
#mpi2prv -f TRACE.mpits -o lackey8$i.prv
#done

#SBATCH --ntasks=12
#SBATCH --cpus-per-task=4
#for i in {1..10};
#do
export OMP_NUM_THREADS=4
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 12 /usr/bin/time -v ./nest $x  &>> n4_12.txt
#mpi2prv -f TRACE.mpits -o lackey12$i.prv
#done

#SBATCH --ntasks=16
#SBATCH --cpus-per-task=3
#for i in {1..10};
#do
export OMP_NUM_THREADS=3
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 16 /usr/bin/time -v  ./nest $x  &>> n4_16.txt
#mpi2prv -f TRACE.mpits -o lackey16$i.prv
#done

#SBATCH --ntasks=24
#SBATCH --cpus-per-task=2
#for i in {1..10};
#do
export OMP_NUM_THREADS=2
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 24 /usr/bin/time -v ./nest $x  &>> n4_24.txt
#mpi2prv -f TRACE.mpits -o lackey24$i.prv
#done

#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#for i in {1..10};
#do
export OMP_NUM_THREADS=1
valgrind --tool=lackey --basic-counts=yes --trace-mem=yes  mpirun -n 48 /usr/bin/time -v ./nest $x &>> n4_48.txt
#mpi2prv -f TRACE.mpits -o lackey48$i.prv
#done

exit
