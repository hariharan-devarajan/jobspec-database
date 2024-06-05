#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=16G
#SBATCH --partition=ampere_gpu
#SBATCH --reservation=structbio2
#SBATCH --time=00:10:00
#SBATCH --job-name=nvt_heat
#SBATCH -o %x.o%j

echo ------------------------------------------------------
echo "${SLURM_JOB_NAME} :: ${SLURM_JOB_ID}"
echo ------------------------------------------------------
echo SLURM: Job submitted from ${SLURM_SUBMIT_HOST}:${SLURM_SUBMIT_DIR}
echo SLURM: Executing queue is $SLURM_JOB_PARTITION
echo SLURM: Job running on nodes $SLURM_JOB_NODELIST
echo SLURM: date: $(date); mytime0=$(date +%s)
echo SLURM: pwd: $(pwd)
echo SLURM: Execution mode is $ENVIRONMENT
echo ------------------------------------------------------

# Load gromacs environment
unalias grep 2> /dev/null
unalias ls 2> /dev/null
check_success() {
    ERRCODE=$?
    [ $ERRCODE -ne 0 ] && echo "[ERROR:${ERRCODE}] $*" && exit $ERRCODE
}
export OMP_PROC_BIND=TRUE
module load GROMACS/2021.3-foss-2021a-CUDA-11.3.1

# Define which simulation to run
WRKDIR="${SLURM_SUBMIT_DIR}"
echo "Goto: $WRKDIR"
cd "$WRKDIR"
check_success "Failure entering directory!"

# Prepare simulation run
CRDFILE=01_min_all.gro
TOPFILE=( $(find ../1_prepare_input/*.top) )
gmx grompp -f 02_nvt_heat.mdp \
           -c "$CRDFILE" \
           -p "$TOPFILE" \
           -r "$CRDFILE" \
           -o 02_nvt_heat.tpr
check_success "Failure occurred executing GMX GROMPP"

# Run the simulation
gmx mdrun -deffnm "02_nvt_heat" -nb gpu -nt $SLURM_CPUS_PER_GPU
check_success "Failure occurred executing GMX MDRUN"

# Quit
mytime1=$(date +%s)
echo "SLURM: Elapsed time: $((mytime1 - mytime0)) seconds"
exit 0
