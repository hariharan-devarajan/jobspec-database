#!/bin/bash

#SBATCH --partition=regular

#SBATCH --account=mp107
#SBATCH --nodes=32
#SBATCH --time=03:30:00
#SBATCH --job-name=pico
#SBATCH --output=out_pico_%j.log

echo Starting slurm script at $(date)

echo -e "\n-----------------------------------------------------------------------"
echo -e "ENVIRONMENT:\n"
env
echo -e "-----------------------------------------------------------------------\n"
echo "PYTHON: $(which python)"
echo "PYTHON VERSION: $(python --version &> /dev/stdout)"
echo ""

pstr=edison-intel
outdir="out_small_satellite_${pstr}"
mkdir -p "${outdir}"

# This script assumes that you are running at NERSC and have already
# loaded the toast module for the correct machine / configuration.

# This should be the same as the --nodes option above
nodes=32
nobs=365

# How many processes are we running per node?  Handle
# the case of a very small simulation.
if [ $nobs -lt 50 ]; then
    node_proc=1
else
    node_proc=8
fi

# Generate the focalplane file if it does not already exist.

detpix=1

fpfile="pico_1.pkl"
if [ ! -e "${fpfile}" ]; then
    srun -n 1 -N 1 bash make_focalplane.sh
fi

# The executable script

ex=$(which toast_satellite_sim.py)
echo "Using ${ex}"

# Scan strategy parameters from a file

parfile="pico_scanning.par"

# Observations


# Map making parameters

nside="1024"

groupnodes=0

# Data distribution parameters.  We are distributing by detector,
# so if our number of processes in a group is larger than the number
# of detectors this is bad.  In that case, set the group size to 
# one, so we have many more groups, each assigned

if [ ${node_proc} -gt ${detpix} ]; then
    groupsize=1
else
    groupsize=$(( node_proc * groupnodes ))
fi

# The commandline

com="${ex} @${parfile} \
--groupsize ${groupsize} \
--fp ${fpfile} \
--nside ${nside} \
--numobs ${nobs} \
--madam \
--baseline 60.0 \
--outdir ${outdir}/out \
"

#--- Hardware configuration ----

# Hyperthread CPUs per physical core
cpu_per_core=2

# Physical cores we are using
node_cores=24

node_thread=$(( node_cores / node_proc ))
node_depth=$(( cpu_per_core * node_thread ))
procs=$(( nodes * node_proc ))

export OMP_NUM_THREADS=${node_thread}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Set TMPDIR to be on the ramdisk
export TMPDIR=/dev/shm

run="srun --cpu_bind=cores -n ${procs} -N ${nodes} -c ${node_depth}"

echo Calling srun at $(date)

echo "${run} ${com}"
eval ${run} ${com} > "${outdir}/log" 2>&1

echo End slurm script at $(date)

