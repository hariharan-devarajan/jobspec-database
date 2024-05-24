#!/bin/bash

#SBATCH -n 24
#SBATCH -N 1
#SBATCH -t 0-30:00
#SBATCH -p fdr
#SBATCH --mem=32000
#SBATCH --mail-type=ALL

# See SLURM documentation for descriptions of all possible settings.
# Type 'man sbatch' at the command prompt to browse documentation.

# Define GEOS-Chem log file
log="gchp.log"

# Sync all config files with settings in runConfig.sh                           
source runConfig.sh > ${log}

checkpoint_file=OutputDir/gcchem_internal_checkpoint
if [[ -e $checkpoint_file ]]; then
    rm $checkpoint_file
fi

if [ -f cap_restart ]; then
   restart_datetime=$(echo $(cat cap_restart) | sed 's/ /_/g')
   restart_file=$checkpoint_file.restart.${restart_datetime}.nc4
   if [[ ! -e $restart_file ]]; then
      echo "cap_restart exists but restart file does not!"
      exit 72
   fi
else
   nCS=$( grep CS_RES= runConfig.sh | cut -d= -f 2 | awk '{print $1}' )
#   restart_file=initial_GEOSChem_rst.c${nCS}_cam26_standard.nc
   restart_file=initial_GEOSChem_c${nCS}_Hg.nc
fi

if [[ -L GCHP_restart.nc4 ]]; then
   unlink GCHP_restart.nc4
fi
ln -s $restart_file GCHP_restart.nc4

# Only start run if no error
if [[ $? == 0 ]]; then

    # Source your environment file. This requires first setting the gchp.env
    # symbolic link using script setEnvironment in the run directory. 
    # Be sure gchp.env points to the same file for both compilation and run.
    gchp_env=$(readlink -f gchp.env)
    if [ ! -f ${gchp_env} ] 
    then
       echo "ERROR: gchp.env symbolic link is not set!"
       echo "Set symbolic link to env file using setEnvironment.sh."
       echo "Exiting."
       exit 1
    fi
    echo " " >> ${log}
    echo "WARNING: You are using environment settings in ${gchp_env}" >> ${log}
    source ${gchp_env} >> ${log}

    # Use SLURM to distribute tasks across nodes
    NX=$( grep NX GCHP.rc | awk '{print $2}' )
    NY=$( grep NY GCHP.rc | awk '{print $2}' )
    coreCount=$(( ${NX} * ${NY} ))
    planeCount=$(( ${coreCount} / ${SLURM_NNODES} ))
    if [[ $(( ${coreCount} % ${SLURM_NNODES} )) > 0 ]]; then
	${planeCount}=$(( ${planeCount} + 1 ))
    fi
    echo "# of CPUs : ${coreCount}" >> ${log}
    echo "# of nodes: ${SLURM_NNODES}" >> ${log}
    echo "-m plane  : ${planeCount}" >> ${log}
    echo ' ' >> ${log}

    # Run the simulation
    echo '===> Run started at' `date` >> ${log}
#    time srun -n ${coreCount} -N ${SLURM_NNODES} -m plane=${planeCount} --mpi=pmix ./gchp >> ${log}
    time mpirun -n ${coreCount} -N ${planeCount} ./gchp >> ${log}
    echo '===> Run ended at' `date` >> ${log}

    # Rename the restart (checkpoint) file to include datetime
    if [ -f cap_restart ]; then
       restart_datetime=$(echo $(cat cap_restart) | sed 's/ /_/g')
       mv OutputDir/gcchem_internal_checkpoint OutputDir/gcchem_internal_checkpoint.restart.${restart_datetime}.nc4
    fi

else
    cat ${log}
fi

exit 0

