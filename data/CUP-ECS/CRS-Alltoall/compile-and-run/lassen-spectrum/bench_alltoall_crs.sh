#!/bin/bash
#BSUB -J alltoall_crs
#BSUB -e alltoall_crs.%J.err
#BSUB -o alltoall_crs.%J.out
##BSUB -nnodes 32
#BSUB -nnodes 2
##BSUB -q pbatch
#BSUB -q pdebug
#BSUB -W 00:15
#BSUB -G unm

moule load valgrind

source ./vars.sh

cd ${CRS_DIR}/mpi_advance/build_lassen/benchmarks
echo ${CRS_DIR}/mpi_advance/build_lassen/benchmarks
folder=${CRS_DIR}/benchmark_mats
echo ${folder}

for mat in delaunay_n22.pm dielFilterV2clx.pm germany_osm.pm human_gene1.pm NLR.pm
do
    echo $folder/$mat
    for (( nodes = 2; nodes <= 32; nodes*=2 ));
    do
        ls
        echo "jsrun -a36 -c36 -r1 -n$nodes valgrind ./alltoall_crs $folder/$mat"
        jsrun -a36 -c36 -r1 -n$nodes valgrind ./alltoall_crs $folder/$mat
        break
    done
    break
done


