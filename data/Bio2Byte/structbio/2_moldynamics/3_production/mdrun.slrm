#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=12G
#SBATCH --partition=pascal_gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=npt_pme
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
lastTPR=$(ls npt_pme_????.tpr 2> /dev/null | sed '$!d')
if [ ! -f "$lastTPR" ]; then
    # Start a new simulation
    iter=1
    nextTPR=npt_pme_0001.tpr
    gmx grompp \
        -f npt_pme.mdp \
        -p ../1_prepare_input/*.top \
        -c $(ls ../2_equilibration/*.gro | sed '$!d') \
        -t $(ls ../2_equilibration/*.cpt | sed '$!d') \
        -o "$nextTPR"
    check_success "Failure occurred executing GMX GROMPP"

    # Run MDRUN
    gmx mdrun \
        -deffnm "npt_pme" \
        -s "$nextTPR" \
        -nt $SLURM_CPUS_PER_GPU \
        -nb gpu
    check_success "Failure occurred executing GMX MDRUN"

else
    # Extend simulation by 100 ns
    iter=$(grep -oE '[1-9][0-9]*' <<<$lastTPR)
    ((iter++))
    nextTPR="$(printf 'npt_pme_%04d.tpr' $iter)"
    gmx convert-tpr -s "$lastTPR" -o "$nextTPR" -extend 100000
    check_success "Failure occurred executing GMX CONVERT-TPR"
    
    # Run MDRUN
    gmx mdrun \
        -deffnm "npt_pme" \
        -s "$nextTPR" \
        -cpi "npt_pme.cpt" \
        -append \
        -nt $SLURM_CPUS_PER_GPU \
        -nb gpu
    check_success "Failure occurred executing GMX MDRUN"
fi

# Quit
mytime1=$(date +%s)
echo "SLURM: Elapsed time: $((mytime1 - mytime0)) seconds"
exit 0
