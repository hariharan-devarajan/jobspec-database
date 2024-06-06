#!/bin/bash

### LSF syntax
#BSUB -nnodes 2                   #number of nodes
#BSUB -W 720                      #walltime in minutes
#BSUB -e log.%J.err               #stderr
#BSUB -o log.%J.out               #stdout
#BSUB -J gpu-benchmark            #name of job
#BSUB -q pbatch                   #queue to use

cd ~/CUP-ECS/ping-pong-gpu/build-hpctoolkit/
source mods

export PINGPONGBIN=$(realpath ~/CUP-ECS/ping-pong-gpu/build-hpctoolkit/ping_pong)
export TESTDIR=/p/gpfs1/haskins8/hpctoolkit-gcc8
#export TESTDIR=../fiesta-tests/3D_Expansion_gpu-pack
#export ENV=./env-files/spectrum-mpi-lassen
#export TAG=poster-hpc-toolkit
export OUTDIR=$(realpath /usr/workspace/haskins8/hpctoolkit-gcc8)
mkdir -p ${TESTDIR}
mkdir -p ${OUTDIR}

#. ${ENV}
#spack load hpctoolkit

#export TESTDIRCOPY=~/gpfs/$(basename ${TESTDIR})_$(basename ${ENV})_$(date --iso-8601=seconds)_${TAG}
cp  ${PINGPONGBIN} ${TESTDIR}
cd ${TESTDIR}
#cp ${PINGPONGBIN} .
export PINGPONGBIN=$(realpath ./$(basename ${PINGPONGBIN}))

#for i in 640; do
for i in 10 20 40 80 160 320 640; do
#for i in 10 20; do
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements0 ${PINGPONGBIN} $i 1000 4 0 0 #--kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements1 ${PINGPONGBIN} $i 1000 4 1 0 #--kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements2 ${PINGPONGBIN} $i 1000 4 2 0 #--kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements3 ${PINGPONGBIN} $i 1000 4 0 1 #--kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements4 ${PINGPONGBIN} $i 1000 4 1 1 #--kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements5 ${PINGPONGBIN} $i 1000 4 2 1 #--kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements6 ${PINGPONGBIN} $i 1000 4 0 2 #--kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements7 ${PINGPONGBIN} $i 1000 4 1 2 #--kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -o hpctoolkit-measurements8 ${PINGPONGBIN} $i 1000 4 2 2 #--kokkos-num-devices=2
done

hpcstruct -o bin.struct ${PINGPONGBIN}
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements0
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements1
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements2
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements3
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements4
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements5
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements6
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements7
lrun -N 2 -T 1 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/CUP-ECS/ping-pong-gpu/src ./hpctoolkit-measurements8

rm -f *sol-*h5*
rm -f *sol-*xmf
rm -f *sol-*lock

mv ${TESTDIR} ${OUTDIR}
