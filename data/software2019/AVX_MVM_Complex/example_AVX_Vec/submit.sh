#!/bin/bash

#SBATCH -J m_2501_3500          # Job name
#SBATCH --partition test        # Job queue
#SBATCH -o job.%j.out           # Name of stdout output file 
#SBATCH -N 1                    # Total number of nodes requested
#SBATCH -n 1                    # Total number of mpi tasks requested
#SBATCH -t 72:00:00             # Run time (hh:mm:ss) - 1.5 hours

source /opt/ohpc/pub/oneAPI/setvars.sh
echo "cat /proc/sys/kernel/perf_event_paranoid"
cat /proc/sys/kernel/perf_event_paranoid

export I_MPI_DEBUG=5 
unset MPI_DSM_OFF                 #Turns off nonuniform memory access (NUMA) optimization in the MPI library.
export MPI_DSM_VERBOSE=1 
export MPI_SHARED_VERBOSE=1 
export MPI_MEMMAP_VERBOSE=1 

# specify number of OMP threads
export OMP_NUM_THREADS=4
# enable thread binding and print out info on thread affinity
export OMP_DISPLAY_ENV=true # turns on display of OMP's internal control variables
export OMP_DISPLAY_AFFINITY=true # display the affinity of each OMP thread
export OMP_AFFINITY_FORMAT="Thread Affinity: %0.3L %.8n %.15{thread_affinity} %.12H"

export OMP_PROC_BIND=close 
export OMP_PLACES=cores #bind each thread to a core


# module load valgrind
# Launch MPI-based executable

  #echo "Number of Threads = " $OMP_NUM_THREADS
  
  #icc xandar_openmp.c -o test1 -qopenmp
  #icc avx_complex_vec.c -o test -O3 -march=core-avx2 -mtune=core-avx2 -no-multibyte-chars
  #icc avx_complex_vec.c -o test -O3 -march=core-avx2 -mtune=core-avx2 -no-multibyte-chars 
  #icc avx_complex_vec.c -o test -O3 -march=core-avx2 -mtune=core-avx2 -no-multibyte-chars

#valgrind --tool=cachegrind --LL=41943040,20,64 ./avx_complex_vec_macro
#valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes --collect-atstart=no   ./avx_complex_vec_align_load  
#valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes --collect-atstart=no   ./avx_complex_vec_align_loadu  
#valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes --collect-atstart=no   ./avx_complex_vec_unalign_loadu    

./avx_complex_vec
# ./avx_complex_vec_align_load 
#./avx_complex_vec_align_load_vasilis 
#./avx_complex_vec_align_loadu
#./avx_complex_vec_unalign_loadu
# ./avx2_benchmark 10

#Without the flag: -no-multibyte-chars, the following error occurs: "Catastrophic error: could not set locale "" to allow processing of multibyte characters"

#Command to run jobs on reserved nodes
#sbatch --reservation=mrahman_13 -q test submit_measure_Dphi.mpi

#valgrind --tool=cachegrind ./p --LL=size,assoc,64 (cache size, associativity and line size)where size and line_size are measured in bytes. --LL=41943040,20,64
#LLC Size: 40960 KB
# Print associativity: cat /sys/devices/system/cpu/cpu0/cache/index3/ways_of_associativity 
#cg_annotate --auto=yes cachegrind.out.46849
#callgrind_annotate --auto=yes callgrind.out.46849