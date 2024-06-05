#!/bin/bash
#! SBATCH
#SBATCH -J frb
#SBATCH -A ACEDO-SL2-CPU
#SBATCH -p icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=1:00:00
#SBATCH --mail-type=FAIL
##SBATCH --no-requeue
##SBATCH --qos=INTR

#! FOR ARRAY JOBS
#SBATCH --array=1-20
# PUT $SLURM_ARRAY_TASK_ID wherever you want the array number
# %A means slurm job ID and %a means array index
# ! in python:
# ! import os
# ! slurm_job_id = os.environ.get('SLURM_ARRAY_TASK_ID')

# Set output and error files
#SBATCH -o slurm-files/%A_%a.out

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! ENV
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl
module load python-3.9.6-gcc-5.4.0-sbr552h
source "$SLURM_SUBMIT_DIR/venv/bin/activate"
application="python"

#! FILE TO RUN
options="$SLURM_SUBMIT_DIR/main.py"

#! WORK DIR
workdir="$SLURM_SUBMIT_DIR"

#! MPI stuff
export OMP_NUM_THREADS=1
np=$[${numnodes}*${mpi_tasks_per_node}]
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets
CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

# Ensure slurm-files directory exists
mkdir -p slurm-files

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

