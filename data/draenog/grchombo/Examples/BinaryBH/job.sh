#!/bin/bash
#COBALT -t 8
#COBALT -n 1
#COBALT --attrs mcdram=cache:numa=quad
#COBALT -A InSituVis2018
echo "Starting Cobalt job script"

pwd

set -x
echo "Starting Cobalt job script"
export n_nodes=$COBALT_JOBSIZE
export n_mpi_ranks_per_node=8
export n_mpi_ranks=$(($n_nodes * $n_mpi_ranks_per_node))
export n_openmp_threads_per_rank=8
export n_hyperthreads_per_core=1
export n_hyperthreads_skipped_between_ranks=2
env | sort
which python
ldd ./GRChombo_BinaryBH3d.Linux.64.CC.ftn.OPTHIGH.MPI.OPENMPCC.ex
aprun -n $n_mpi_ranks -N $n_mpi_ranks_per_node \
  --env OMP_NUM_THREADS=$n_openmp_threads_per_rank -cc depth \
  -d $n_hyperthreads_skipped_between_ranks \
  -j $n_hyperthreads_per_core \
  -e LD_LIBRARY_PATH=/soft/visualization/paraview/v5.5.2.0_icc.17.0.4.196_mpich.7.7.0/lib:/opt/python/2.7.13.1/lib:$LD_LIBRARY_PATH \
  -e PYTHONPATH=/soft/visualization/paraview/v5.5.2/lib/python2.7/site-packages:/soft/visualization/paraview/v5.5.2/lib/python2.7/site-packages/vtkmodules \
  -e GALLIUM_DRIVER=swr \
  ./GRChombo_BinaryBH3d.Linux.64.CC.ftn.OPTHIGH.MPI.OPENMPCC.ex mybh.txt
