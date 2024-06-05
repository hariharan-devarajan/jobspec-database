#!/bin/bash

#SBATCH --job-name=apoa1benchmark
#SBATCH --nodes=1
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
declare -r NAMD2_BINARY=__NAMD2BINARY__
declare -r NAMD3_BINARY=__NAMD3BINARY__

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
#   SBATCH --nodes=1                                 -
#                                                    -
#-----------------------------------------------------
declare -ri number_physical_nodes=${SLURM_JOB_NUM_NODES}

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
cp ${NAMD2_BINARY} ${SCRATCH_DIR}/
cp ${NAMD3_BINARY} ${SCRATCH_DIR}/

# ----------------------------------------------------
# Copy all input files to the scratch directory.     -
#                                                    -
# ----------------------------------------------------
input_files=( "apoa1.pdb"
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
echo 'charmrun ./namd2 +p2 ./apoa1.namd 1> __STDOUT__ 2> __STDERR__' 
charmrun ./namd3 +p2 ./apoa1.namd  1> __STDOUT__ 2> __STDERR__

#-----------------------------------------------------
# Copy all files back to the results directory.      -
#                                                    -
#-----------------------------------------------------
cp -rf ${SCRATCH_DIR}/* ${NAMD_RESULTS_DIR}/ 
