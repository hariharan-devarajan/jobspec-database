#!/bin/bash

#SBATCH -A CFD162
#SBATCH -J val2cumulative
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

#####
# N = number of nodes
# --ntasks-per-node = gpu_ranks_per_node
#####

##### salloc --account=CFD162 --time=2:00:00 -N32 -n256 --ntasks-per-gpu=1 --gpu-bind=closest --mail-type=ALL --mail-user=kevmoor@sandia.gov --qos=debug

##################
nodes_requested=$SLURM_JOB_NUM_NODES
ranks_requested=$SLURM_NTASKS
##################

export rocm_version=5.4.3

#module purge
module load PrgEnv-amd
module load amd/${rocm_version}
module load craype-accel-amd-gfx90a
module load cray-mpich

export FI_MR_CACHE_MONITOR=memhooks
export FI_CXI_RX_MATCH_MODE=software
#export HIP_LAUNCH_BLOCKING=1

export SPACK_MANAGER=/lustre/orion/cfd162/proj-shared/kevmoor/spack-manager
source ${SPACK_MANAGER}/start.sh
spack-start
spack env activate -d ${SPACK_MANAGER}/environments/naluGPUOverset
spack load nalu-wind

#export mytimestamp=$(grep "Current Time:" test_pitching_test9.out | tail -n 1 | sed -n 's/.*Current Time: \(.*\)/\1/p')

#sed -i "s/restart_time: [0-9.]\+/restart_time: ${mytimestamp}/" airfoil_osc_overset_periodic.yaml


srun -u -N32 -n256 --ntasks-per-gpu=1 --gpu-bind=closest naluX -i Validation_AOA2_1_Variation8_2_Freq075Cumulative.yaml -o val2cumulative.out

# grids=../ffa_w3_360/static/
# echo ${grids}
# list_of_cases=($(ls ${grids}))

# for idx in {0..58};

#    do

#       echo " "
#       echo "Job: "$idx &
#       echo "Case: "${list_of_cases[$idx]} &
#       echo "Directory contents of " ${list_of_cases[$idx]} &
#       ls -alh ${grids}/${list_of_cases[$idx]} &

#       date &
      
#       echo "Yaml file: " $grids${list_of_cases[$idx]}/*.yaml
#       srun -u -N32 -n256 --ntasks-per-gpu=1 --gpu-bind=closest naluX -i $grids${list_of_cases[$idx]}/*.yaml -o $grids${list_of_cases[$idx]}/log$idx.out &

#       sleep 1

#    done

# wait
