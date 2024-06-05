#!/bin/bash

#SBATCH --job-name=apoa1benchmark
#SBATCH --nodes=2
#SBATCH -A __ACCOUNTID__
#SBATCH --time=00:30:00

# ----------------------------------------------------
# Set the scratch dir.
# 
# ----------------------------------------------------
declare -r SCRATCH_DIR=__SCRATCHSPACE__

# ----------------------------------------------------
# Set the fully qualified path to the NAMD binary.
#
# ----------------------------------------------------
declare -r NAMD_BINARY=__NAMDBINARY__

#-----------------------------------------------------
# Set the fully qualified path to the results        -
# directory.                                         -
#                                                    -
#-----------------------------------------------------
declare -r NAMD_RESULTS_DIR=__NAMDRESULTSDIR__

#-----------------------------------------------------
# Set the fully qualified path the the input files   -
# parent directory.                                  -
#                                                    -
#-----------------------------------------------------
declare -r INPUT_FILES_PARENT_DIR=__APOA1_INPUT_FILES_PARENT_DIR__

#-----------------------------------------------------
# Set the number of physical nodes.                  -
# SLURM_JOB_NUM_NODES is the number of nodes         -
# requested:                                         -
#   SBATCH --nodes=2                                 -
#                                                    -
#-----------------------------------------------------
declare -ri number_physical_nodes=${SLURM_JOB_NUM_NODES}

#-----------------------------------------------------
# A Spock compute node has 4 NUMA domains.           -
#                                                    -         
# NUMA 0: hardware threads 000-015, 064-079 | GPU 0  -
# NUMA 1: hardware threads 016-031, 080-095 | GPU 1  -
# NUMA 2: hardware threads 032-047, 096-111 | GPU 2  -
# NUMA 3: hardware threads 048-063, 112-127 | GPU 3  -
#                                                    -
# We map the charm++ logical node to a NUMA          -
# domain. This implies a maximum of 4 charm++        -
# logical nodes per physical node.                   -
#                                                    -
#-----------------------------------------------------
declare -ri max_number_charm_logical_nodes_per_physical_node=4

#-----------------------------------------------------
# We now set the mapping of PEs and COMM.            -
#                                                    -
# For NUMA 0, we reserve hardware thread 0 for comm. -
# and 1-15 for PE.                                   -
#
# For NUMA 1, we reserve hardware thread 16 for comm.-
# and 17-31 for PE.                                  -
# 
# For NUMA 2, we reserve hardware thread 32 for comm.-
# and 33-47 for PE.                                  -
#                                                    -
# For NUMA 3, we reserve hardware thread 48 for comm.-
# and 49-63 for PE.                                  -
#                                                    -
#-----------------------------------------------------
declare -r pe_com_map="+commap 0,16,32,48 +pemap 1-15+16+32+48"

# ----------------------------------------------------
# Copy the NAMD binary to the scratch directory.
#
# ----------------------------------------------------
cp ${NAMD_BINARY} ${SCRATCH_DIR}/

# ----------------------------------------------------
# Copy all input files to the scratch directory.     -
#                                                    -
# ----------------------------------------------------
input_files=( "apoa1.namd" 
              "apoa1.pdb"
              "apoa1.psf"
              "par_all22_popc.xplor"
              "par_all22_prot_lipid.xplor" ) 

for tmp_input_file in "${input_files[@]}";do
    cp -f "${INPUT_FILES_PARENT_DIR}/${tmp_input_file}" "${SCRATCH_DIR}"
done

#-----------------------------------------------------
# Now run namd with charmrun.                        -
#                                                    -
#-----------------------------------------------------
cd ${SCRATCH_DIR}/
# srun -n 4 -N 2 ./namd2 ++p 2 ++ppn 2 ${pe_com_map} ./apoa1.namd
declare -r nm_charm_process=2
declare -r charm_process_per_node=2
declare -r max_tasks_per_core=2
declare -r ntasks=4
echo 'charmrun ++mpiexec ++remote-shell "srun" ./namd2 ++n 6 ++ppn 2 ${pe_com_map} ./apoa1.namd ' 
charmrun ++mpiexec ++remote-shell "srun" ./namd2 ++n 6 ++ppn 2 ${pe_com_map} ./apoa1.namd  

#-----------------------------------------------------
# Copy all files back to the results directory.      -
#                                                    -
#-----------------------------------------------------
cp -rf ${SCRATCH_DIR}/* ${NAMD_RESULTS_DIR}/ 
