#!/usr/bin/env bash
#
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --nodelist=cn61[4-7]
#SBATCH --partition=skylake-gold
#SBATCH --nodes=4

#echo "skylake: $(hostname)"

echo ${SLURM_JOB_NODELIST}
module unload
module load gcc/7.3.0 cmake openmpi/3.1.3-gcc_7.3.0 likwid
srun -w ${SLURM_JOB_NODELIST} -n 4 hostname

# export OMP_NUM_THREADS=1

# RESULTS="${HOME}/memsys19-skylake/results"
# mkdir -p ${RESULTS}

# cd /home/jlee/memsys19-skylake/vpic/skylake

# # vpic weak scaling for skylake modes, nppc in lpic_weak should be 32768
# # export MACRO="USE_OPTANE"
# # bin/vpic lpi_weak.cxx
# # for p in 1 8 16 32
# # do
# #     mpirun -np $p likwid-perfctr -g ENERGY ./lpi_weak.Linux   &> ${RESULTS}/vpic_weak_scaling.skylake.$p.ENERGY
# #     mpirun -np $p likwid-perfctr -g FLOPS_DP ./lpi_weak.Linux &> ${RESULTS}/vpic_weak_scaling.skylake.$p.FLOPS_DP
# #     mpirun -np $p likwid-perfctr -g MEM ./lpi_weak.Linux      &> ${RESULTS}/vpic_weak_scaling.skylake.$p.MEM
# #     mpirun -np $p likwid-perfctr -g L3CACHE ./lpi_weak.Linux  &> ${RESULTS}/vpic_weak_scaling.skylake.$p.L3CACHE
# # done

# export MACRO="USE_DRAM"
# bin/vpic lpi_weak.cxx
# for p in 1 8 16 32
# do
#     mpirun -np $p likwid-perfctr -g ENERGY ./lpi_weak.Linux   &> ${RESULTS}/vpic_weak_scaling.DRAM.$p.ENERGY
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./lpi_weak.Linux &> ${RESULTS}/vpic_weak_scaling.DRAM.$p.FLOPS_DP
#     mpirun -np $p likwid-perfctr -g MEM ./lpi_weak.Linux      &> ${RESULTS}/vpic_weak_scaling.DRAM.$p.MEM
#     mpirun -np $p likwid-perfctr -g L3CACHE ./lpi_weak.Linux  &> ${RESULTS}/vpic_weak_scaling.DRAM.$p.L3CACHE
# done

# # # vpic strong scaling for skylake modes
# # export MACRO="USE_OPTANE"
# # for npcc in 1024 2048 4096 32768
# # do
# #     bin/vpic strongscaling-${npcc}.cxx
# # done

# # declare -A npccranks
# # npccranks[1024]=32
# # npccranks[2048]=16
# # npccranks[4096]=8
# # npccranks[32768]=1

# # for i in "${!npccranks[@]}"
# # do
# #     mpirun -np "${npccranks[$i]}" likwid-perfctr -g ENERGY   ./strongscaling-"$i".Linux   &> "${RESULTS}/vpic_strong_scaling.skylake.$i.${npccranks[$i]}.ENERGY"
# #     mpirun -np "${npccranks[$i]}" likwid-perfctr -g FLOPS_DP ./strongscaling-"$i".Linux   &> "${RESULTS}/vpic_strong_scaling.skylake.$i.${npccranks[$i]}.FLOPS_DP"
# #     mpirun -np "${npccranks[$i]}" likwid-perfctr -g MEM      ./strongscaling-"$i".Linux   &> "${RESULTS}/vpic_strong_scaling.skylake.$i.${npccranks[$i]}.MEM"
# #     mpirun -np "${npccranks[$i]}" likwid-perfctr -g L3CACHE  ./strongscaling-"$i".Linux   &> "${RESULTS}/vpic_strong_scaling.skylake.$i.${npccranks[$i]}.L3CACHE"
# # done

# export MACRO="USE_DRAM"
# for npcc in 1024 2048 4096 32768
# do
#     bin/vpic strongscaling-${npcc}.cxx
# done

# for i in "${!npccranks[@]}"
# do
#     mpirun -np "${npccranks[$i]}" likwid-perfctr -g ENERGY   ./strongscaling-"$i".Linux   &> "${RESULTS}/vpic_strong_scaling.DRAM.$i.${npccranks[$i]}.ENERGY"
#     mpirun -np "${npccranks[$i]}" likwid-perfctr -g FLOPS_DP ./strongscaling-"$i".Linux   &> "${RESULTS}/vpic_strong_scaling.DRAM.$i.${npccranks[$i]}.FLOPS_DP"
#     mpirun -np "${npccranks[$i]}" likwid-perfctr -g MEM      ./strongscaling-"$i".Linux   &> "${RESULTS}/vpic_strong_scaling.DRAM.$i.${npccranks[$i]}.MEM"
#     mpirun -np "${npccranks[$i]}" likwid-perfctr -g L3CACHE  ./strongscaling-"$i".Linux   &> "${RESULTS}/vpic_strong_scaling.DRAM.$i.${npccranks[$i]}.L3CACHE"
# done

# # SNAP strong scaling
# # process count increases 1,8,16,32, nx, ny, nz stay at 256
# cd /home/jlee/memsys19-skylake/SNAP/ports/snap-c
# make -j clean
# make -j
# for p in 1 8 16 32
# do
#     # mpirun -np $p likwid-perfctr -g ENERGY   ./snap --fi ${p}_strong --fo ${RESULTS}/snap_strong_scaling.skylake.$p.ENERGY SKYLAKE &> ${RESULTS}/snap_strong_scaling.skylake.$p.ENERGY.likwid
#     # mpirun -np $p likwid-perfctr -g FLOPS_DP ./snap --fi ${p}_strong --fo ${RESULTS}/snap_strong_scaling.skylake.$p.FLOPS_DP SKYLAKE &> ${RESULTS}/snap_strong_scaling.skylake.$p.FLOPS_DP.likwid
#     # mpirun -np $p likwid-perfctr -g MEM      ./snap --fi ${p}_strong --fo ${RESULTS}/snap_strong_scaling.skylake.$p.MEM SKYLAKE &> ${RESULTS}/snap_strong_scaling.skylake.$p.MEM.likwid
#     # mpirun -np $p likwid-perfctr -g L3CACHE  ./snap --fi ${p}_strong --fo ${RESULTS}/snap_strong_scaling.skylake.$p.L3CACHE SKYLAKE &> ${RESULTS}/snap_strong_scaling.skylake.$p.L3CACHE.likwid

#     mpirun -np $p likwid-perfctr -g ENERGY   ./snap --fi ${p}_strong --fo ${RESULTS}/snap_strong_scaling.DRAM.$p.ENERGY DRAM    &> ${RESULTS}/snap_strong_scaling.DRAM.$p.ENERGY.likwid
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./snap --fi ${p}_strong --fo ${RESULTS}/snap_strong_scaling.DRAM.$p.FLOPS_DP DRAM  &> ${RESULTS}/snap_strong_scaling.DRAM.$p.FLOPS_DP.likwid
#     mpirun -np $p likwid-perfctr -g MEM      ./snap --fi ${p}_strong --fo ${RESULTS}/snap_strong_scaling.DRAM.$p.MEM DRAM       &> ${RESULTS}/snap_strong_scaling.DRAM.$p.MEM.likwid
#     mpirun -np $p likwid-perfctr -g L3CACHE  ./snap --fi ${p}_strong --fo ${RESULTS}/snap_strong_scaling.DRAM.$p.L3CACHE DRAM   &> ${RESULTS}/snap_strong_scaling.DRAM.$p.L3CACHE.likwid
# done

# # SNAP weak scaling
# # process count increases nx,ny,nz --> 8 64 128 256
# for p in 1 8 16 32
# do
#     # mpirun -np $p likwid-perfctr -g ENERGY   ./snap --fi ${p}_weak --fo ${RESULTS}/snap_weak_scaling.skylake.$p.ENERGY SKYLAKE &> ${RESULTS}/snap_weak_scaling.skylake.$p.ENERGY.likwid
#     # mpirun -np $p likwid-perfctr -g FLOPS_DP ./snap --fi ${p}_weak --fo ${RESULTS}/snap_weak_scaling.skylake.$p.FLOPS_DP SKYLAKE &> ${RESULTS}/snap_weak_scaling.skylake.$p.FLOPS_DP.likwid
#     # mpirun -np $p likwid-perfctr -g MEM      ./snap --fi ${p}_weak --fo ${RESULTS}/snap_weak_scaling.skylake.$p.MEM SKYLAKE &> ${RESULTS}/snap_weak_scaling.skylake.$p.MEM.likwid
#     # mpirun -np $p likwid-perfctr -g L3CACHE  ./snap --fi ${p}_weak --fo ${RESULTS}/snap_weak_scaling.skylake.$p.L3CACHE SKYLAKE &> ${RESULTS}/snap_weak_scaling.skylake.$p.L3CACHE.likwid

#     mpirun -np $p likwid-perfctr -g ENERGY   ./snap --fi ${p}_weak --fo ${RESULTS}/snap_weak_scaling.DRAM.$p.ENERGY DRAM    &> ${RESULTS}/snap_weak_scaling.DRAM.$p.ENERGY.likwid
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./snap --fi ${p}_weak --fo ${RESULTS}/snap_weak_scaling.DRAM.$p.FLOPS_DP DRAM  &> ${RESULTS}/snap_weak_scaling.DRAM.$p.FLOPS_DP.likwid
#     mpirun -np $p likwid-perfctr -g MEM      ./snap --fi ${p}_weak --fo ${RESULTS}/snap_weak_scaling.DRAM.$p.MEM DRAM       &> ${RESULTS}/snap_weak_scaling.DRAM.$p.MEM.likwid
#     mpirun -np $p likwid-perfctr -g L3CACHE  ./snap --fi ${p}_weak --fo ${RESULTS}/snap_weak_scaling.DRAM.$p.L3CACHE DRAM   &> ${RESULTS}/snap_weak_scaling.DRAM.$p.L3CACHE.likwid
# done

# AMG weakscaling
# process 1, 8, 16, 32; n = 512
# cd /home/jlee/memsys19-skylake/AMG/test
# make clean
# make -j
# for p in 1 8 16 32
# do
#     # mpirun -np $p likwid-perfctr -g ENERGY   ./amg -n 512 512 512 OPTANE &> ${RESULTS}/amg_weak_scaling.skylake.$p.ENERGY
#     # mpirun -np $p likwid-perfctr -g FLOPS_DP ./amg -n 512 512 512 OPTANE &> ${RESULTS}/amg_weak_scaling.skylake.$p.FLOPS_DP
#     # mpirun -np $p likwid-perfctr -g MEM      ./amg -n 512 512 512 OPTANE &> ${RESULTS}/amg_weak_scaling.skylake.$p.MEM
#     # mpirun -np $p likwid-perfctr -g L3CACHE  ./amg -n 512 512 512 OPTANE &> ${RESULTS}/amg_weak_scaling.skylake.$p.L3CACHE

#     mpirun -np $p likwid-perfctr -g ENERGY   ./amg -n 512 512 512 DRAM &> ${RESULTS}/amg_weak_scaling.DRAM.$p.ENERGY
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./amg -n 512 512 512 DRAM &> ${RESULTS}/amg_weak_scaling.DRAM.$p.FLOPS_DP
#     mpirun -np $p likwid-perfctr -g MEM      ./amg -n 512 512 512 DRAM &> ${RESULTS}/amg_weak_scaling.DRAM.$p.MEM
#     mpirun -np $p likwid-perfctr -g L3CACHE  ./amg -n 512 512 512 DRAM &> ${RESULTS}/amg_weak_scaling.DRAM.$p.L3CACHE
# done

# # AMG strong scaling
# # process 1, 8, 16, 32; n = (512, 512, 512), (256, 256, 256), (128, 256, 256), (128, 128 256)

# declare -A N
# N[1]="512 512 512"
# N[8]="256 256 256"
# N[16]="128 256 256"
# N[32]="128 128 256"

# declare -A P
# P[1]="1 1 1"
# P[8]="2 2 2"
# P[16]="4 2 2"
# P[32]="4 4 2"


# for p in "${!n[@]}"
# do
#     # mpirun -np $p likwid-perfctr -g ENERGY   ./amg -n ${n[$p]} OPTANE &> ${RESULTS}/amg_strong_scaling.skylake.$p.ENERGY
#     # mpirun -np $p likwid-perfctr -g FLOPS_DP ./amg -n ${n[$p]} OPTANE &> ${RESULTS}/amg_strong_scaling.skylake.$p.FLOPS_DP
#     # mpirun -np $p likwid-perfctr -g MEM      ./amg -n ${n[$p]} OPTANE &> ${RESULTS}/amg_strong_scaling.skylake.$p.MEM
#     # mpirun -np $p likwid-perfctr -g L3CACHE  ./amg -n ${n[$p]} OPTANE &> ${RESULTS}/amg_strong_scaling.skylake.$p.L3CACHE

#     mpirun -np $p likwid-perfctr -g ENERGY   ./amg -n ${N[$p]} -p ${P[$p]} DRAM &> ${RESULTS}/amg_strong_scaling.DRAM.$p.ENERGY
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./amg -n ${N[$p]} -p ${P[$p]} DRAM &> ${RESULTS}/amg_strong_scaling.DRAM.$p.FLOPS_DP
#     mpirun -np $p likwid-perfctr -g MEM      ./amg -n ${N[$p]} -p ${P[$p]} DRAM &> ${RESULTS}/amg_strong_scaling.DRAM.$p.MEM
#     mpirun -np $p likwid-perfctr -g L3CACHE  ./amg -n ${N[$p]} -p ${P[$p]} DRAM &> ${RESULTS}/amg_strong_scaling.DRAM.$p.L3CACHE
# done

# # DGEMM weak scaling
# # process 4, 8, 16, 32; N = 64000
# cd /home/jlee/memsys19-skylake/hpcc-1.5.0
# make arch=skylake -j clean
# make arch=skylake -j
# for p in 4 8 16 32
# do
#     # mpirun -np $p likwid-perfctr -g ENERGY   ./hpcc hpccinf.txt OPTANE &> ${RESULTS}/hpcc_strong_scaling.skylake.$p.ENERGY
#     # mpirun -np $p likwid-perfctr -g FLOPS_DP ./hpcc hpccinf.txt OPTANE &> ${RESULTS}/hpcc_strong_scaling.skylake.$p.FLOPS_DP
#     # mpirun -np $p likwid-perfctr -g MEM      ./hpcc hpccinf.txt OPTANE &> ${RESULTS}/hpcc_strong_scaling.skylake.$p.MEM
#     # mpirun -np $p likwid-perfctr -g L3CACHE  ./hpcc hpccinf.txt OPTANE &> ${RESULTS}/hpcc_strong_scaling.skylake.$p.L3CACHE

#     mpirun -np $p likwid-perfctr -g ENERGY   ./hpcc hpccinf.txt &> ${RESULTS}/hpcc_strong_scaling.DRAM.$p.ENERGY
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./hpcc hpccinf.txt &> ${RESULTS}/hpcc_strong_scaling.DRAM.$p.FLOPS_DP
#     mpirun -np $p likwid-perfctr -g MEM      ./hpcc hpccinf.txt &> ${RESULTS}/hpcc_strong_scaling.DRAM.$p.MEM
#     mpirun -np $p likwid-perfctr -g L3CACHE  ./hpcc hpccinf.txt &> ${RESULTS}/hpcc_strong_scaling.DRAM.$p.L3CACHE
# done

# # DGEMM strong scaling
# # process 4, 8, 16, 32; N = 64K 32K 16K 8K
# unset n
# declare -A n
# n[4]=8000
# n[8]=16000
# n[16]=32000
# n[32]=64000

# for p in "${!n[@]}"
# do
#     # mpirun -np $p likwid-perfctr -g ENERGY   ./hpcc $p OPTANE &> ${RESULTS}/hpcc_weak_scaling.skylake.$p.ENERGY
#     # mpirun -np $p likwid-perfctr -g FLOPS_DP ./hpcc $p OPTANE &> ${RESULTS}/hpcc_weak_scaling.skylake.$p.FLOPS_DP
#     # mpirun -np $p likwid-perfctr -g MEM      ./hpcc $p OPTANE &> ${RESULTS}/hpcc_weak_scaling.skylake.$p.MEM
#     # mpirun -np $p likwid-perfctr -g L3CACHE  ./hpcc $p OPTANE &> ${RESULTS}/hpcc_weak_scaling.skylake.$p.L3CACHE

#     mpirun -np $p likwid-perfctr -g ENERGY   ./hpcc $p &> ${RESULTS}/hpcc_weak_scaling.DRAM.$p.ENERGY
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./hpcc $p &> ${RESULTS}/hpcc_weak_scaling.DRAM.$p.FLOPS_DP
#     mpirun -np $p likwid-perfctr -g MEM      ./hpcc $p &> ${RESULTS}/hpcc_weak_scaling.DRAM.$p.MEM
#     mpirun -np $p likwid-perfctr -g L3CACHE  ./hpcc $p &> ${RESULTS}/hpcc_weak_scaling.DRAM.$p.L3CACHE
# done

# # LULESH weak scaling
# # processes 1, 8, 27, 64
# cd /home/jlee/memsys19-skylake/LULESH
# make -j clean
# make -j
# for p in 1 8 27
# do
#     # mpirun -np $p likwid-perfctr -g ENERGY   ./lulesh2.0 -s 64 OPTANE &> ${RESULTS}/lulesh_weak_scaling.skylake.$p.ENERGY
#     # mpirun -np $p likwid-perfctr -g FLOPS_DP ./lulesh2.0 -s 64 OPTANE &> ${RESULTS}/lulesh_weak_scaling.skylake.$p.FLOPS_DP
#     # mpirun -np $p likwid-perfctr -g MEM      ./lulesh2.0 -s 64 OPTANE &> ${RESULTS}/lulesh_weak_scaling.skylake.$p.MEM
#     # mpirun -np $p likwid-perfctr -g L3CACHE  ./lulesh2.0 -s 64 OPTANE &> ${RESULTS}/lulesh_weak_scaling.skylake.$p.L3CACHE

#     mpirun -np $p likwid-perfctr -g ENERGY   ./lulesh2.0 -s 64 DRAM &> ${RESULTS}/lulesh_weak_scaling.DRAM.$p.ENERGY
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./lulesh2.0 -s 64 DRAM &> ${RESULTS}/lulesh_weak_scaling.DRAM.$p.FLOPS_DP
#     mpirun -np $p likwid-perfctr -g MEM      ./lulesh2.0 -s 64 DRAM &> ${RESULTS}/lulesh_weak_scaling.DRAM.$p.MEM
#     mpirun -np $p likwid-perfctr -g L3CACHE  ./lulesh2.0 -s 64 DRAM &> ${RESULTS}/lulesh_weak_scaling.DRAM.$p.L3CACHE
# done

# # LULESH strong scaling
# # processes 1, 8, 27, 64; problem size 40, 20, 13, 10

# declare -A procsize
# procsize[1]=96
# procsize[8]=48
# procsize[27]=32
# #procsize[64]=16

# for p in "${!procsize[@]}"
# do
#     # mpirun -np $p likwid-perfctr -g ENERGY   ./lulesh2.0 -s ${procsize[$p]} OPTANE &> ${RESULTS}/lulesh_strong_scaling.skylake.$p.ENERGY
#     # mpirun -np $p likwid-perfctr -g FLOPS_DP ./lulesh2.0 -s ${procsize[$p]} OPTANE &> ${RESULTS}/lulesh_strong_scaling.skylake.$p.FLOPS_DP
#     # mpirun -np $p likwid-perfctr -g MEM      ./lulesh2.0 -s ${procsize[$p]} OPTANE &> ${RESULTS}/lulesh_strong_scaling.skylake.$p.MEM
#     # mpirun -np $p likwid-perfctr -g L3CACHE  ./lulesh2.0 -s ${procsize[$p]} OPTANE &> ${RESULTS}/lulesh_strong_scaling.skylake.$p.L3CACHE

#     mpirun -np $p likwid-perfctr -g ENERGY   ./lulesh2.0 -s ${procsize[$p]} DRAM &> ${RESULTS}/lulesh_strong_scaling.DRAM.$p.ENERGY
#     mpirun -np $p likwid-perfctr -g FLOPS_DP ./lulesh2.0 -s ${procsize[$p]} DRAM &> ${RESULTS}/lulesh_strong_scaling.DRAM.$p.FLOPS_DP
#     mpirun -np $p likwid-perfctr -g MEM      ./lulesh2.0 -s ${procsize[$p]} DRAM &> ${RESULTS}/lulesh_strong_scaling.DRAM.$p.MEM
#     mpirun -np $p likwid-perfctr -g L3CACHE  ./lulesh2.0 -s ${procsize[$p]} DRAM &> ${RESULTS}/lulesh_strong_scaling.DRAM.$p.L3CACHE
# done
