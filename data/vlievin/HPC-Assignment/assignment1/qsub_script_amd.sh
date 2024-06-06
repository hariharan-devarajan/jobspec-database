#!/bin/sh

# embedded options to qsub - start with #PBS
# -- Name of the job ---
#PBS -N HPC_matrix_multipication_amd
# -- specify queue --
#PBS -q hpc
# -- estimated wall clock time (execution time): hh:mm:ss --
#PBS -l walltime=01:30:00
# -- number of processors/cores/nodes (request X node with X cores) --
#PBS -l nodes=1:ppn=32
# -- CPU type --
#PBS -l feature='Opteron6136'
# -- user email address --
#PBS -M s151457@student.dtu.dk
# -- mail notification when begins(b), ends(e) and fails (a=aborted) --
#PBS -m a

# -- run in the current working (submission) directory --
if test X$PBS_ENVIRONMENT = XPBS_BATCH; then cd $PBS_O_WORKDIR; fi

outfile="out_lib_${CPUTYPE}-fast.dat"

NPROCS=`wc -l < "${PBS_NODEFILE}"`
(
        echo == CPUTYPE: $CPUTYPE
        echo == NPROCS: $NPROCS
        echo == PBS_NP: $PBS_NP
        echo == PBS_NODENUM: $PBS_NODENUM
        echo == PBS_NUM_NODES: $PBS_NUM_NODES
        echo == PBS_NUM_PPN: $PBS_NUM_PPN
        echo == PBS_TASKNUM: $PBS_TASKNUM

	# call one or multiple scripts
	./mm-multiplication.sh -o $outfile -1

) >> $outfile

exit 0
