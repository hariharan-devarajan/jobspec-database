#!/bin/bash

### LSF syntax
#BSUB -nnodes 2                   #number of nodes
#BSUB -W 30                       #walltime in minutes
#BSUB -e log.%J.err               #stderr
#BSUB -o log.%J.out               #stdout
#BSUB -J gpu-pack                 #name of job
#BSUB -q pbatch                   #queue to use

cd ~/CUP-ECS/ping-pong-gpu/

export PINGPONGBIN=$(realpath ~/CUP-ECS/ping-pong-gpu/build-mvapich2/ping_pong)
export TESTDIR=/p/gpfs1/haskins8/mvapich2
#export TESTDIR=../fiesta-tests/3D_Expansion_gpu-pack
#export ENV=./env-files/spectrum-mpi-lassen
#export TAG=poster-hpc-toolkit
export OUTDIR=$(realpath /usr/workspace/haskins8/mvapich2-out/)
mkdir -p ${TESTDIR}
mkdir -p ${OUTDIR}

#. ${ENV}
#spack load hpctoolkit

#export TESTDIRCOPY=~/gpfs/$(basename ${TESTDIR})_$(basename ${ENV})_$(date --iso-8601=seconds)_${TAG}
cp  ${PINGPONGBIN} ${TESTDIR}
cd ${TESTDIR}
#cp ${PINGPONGBIN} .
export PINGPONGBIN=$(realpath ./$(basename ${PINGPONGBIN}))

for i in 10 20 40 80 160 320 640; do
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
    lrun -M "-gpu" -N 2 -T 1 -g 1 --gpubind=off hpcrun -t -e CPUTIME@f99 -e gpu=nvidia@f99 -o hpctoolkit-measurements ${PINGPONGBIN} $i 1000 4 0 0 --kokkos-num-devices=2
done
hpcstruct -o bin.struct ${PINGPONGBIN}
#lrun -N16 -T4 hpcprof-mpi --metric-db yes -S ./bin.struct -I ~/workspace/cup-ecs-fiesta/src ./hpctoolkit-measurements

rm -f *sol-*h5*
rm -f *sol-*xmf
rm -f *sol-*lock

mv ${TESTDIR} ${OUTDIR}
