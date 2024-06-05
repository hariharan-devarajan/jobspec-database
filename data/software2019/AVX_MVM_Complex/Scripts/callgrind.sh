#!/bin/bash

#SBATCH -J m_2501_3600            # Job name
#SBATCH --partition test          # Job queue
#SBATCH -o job.%j.out             # Name of stdout output file 
#SBATCH -N 1                      # Total number of nodes requested
#SBATCH -n 1                      # Total number of mpi tasks requested
#SBATCH -t 72:00:00               # Run time (hh:mm:ss) - 1.5 hours

echo "My Valgrind Script Running..."

echo "cat /proc/sys/kernel/perf_event_paranoid"
cat /proc/sys/kernel/perf_event_paranoid

export I_MPI_DEBUG=5 
export OMP_DISPLAY_AFFINITY=TRUE 
unset MPI_DSM_OFF 
export MPI_DSM_VERBOSE=1 
export MPI_SHARED_VERBOSE=1 
export MPI_MEMMAP_VERBOSE=1 


module load valgrind
#rm out_0
#make -j6
export OMP_NUM_THREADS=1
#mpirun [mpi-opts] valgrind [valgrind-opt] ./app [app-args] 
#mpirun -n 1 valgrind --tool=cachegrind --LL=41943040,20,64 ./speed_test_diracoperator -i ./speed_test_diracoperator.in -o out_sf_measure_2501_3501
mpirun -n 1 valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes --collect-atstart=no ./speed_test_diracoperator -i ./speed_test_diracoperator.in -o out_sf_measure_2501_3501
#valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes --collect-atstart=no   ./avx_complex_vec_align_load 
#mpirun -n 1 valgrind --tool=cachegrind ./speed_test_diracoperator -i ./speed_test_diracoperator.in 
#cat out_0 

#valgrind --tool=cachegrind ./p --LL=size,assoc,64 (cache size, associativity and line size)where size and line_size are measured in bytes
#LLC Size: 40960 KB
# Print associativity: cat /sys/devices/system/cpu/cpu0/cache/index3/ways_of_associativity 
#cg_annotate cachegrind.out.46849
#callgrind_annotate --auto=yes callgrind.out.46849

echo "My Script Ended."