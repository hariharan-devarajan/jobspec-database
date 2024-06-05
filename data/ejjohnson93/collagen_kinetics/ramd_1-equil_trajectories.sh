#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job.
#SBATCH --export=ALL
# Define job name
#SBATCH -J trimers_10ns
# Define a standard output file. When the job is running, %u will be replaced by user name,
# %N will be replaced by the name of the node that runs the batch script, and %j will be replaced by job id number.
#SBATCH -o trimers_10ns.%u.%N.%j.out
# Define a standard error file
#SBATCH -e trimers_10ns.%u.%N.%j.err
# Request the partition
#SBATCH -p nodes
# Request the number of nodes
#SBATCH -N 8
# Specify the number of tasks per node
#SBATCH --ntasks-per-node=8
# Specify the number of tasks
# Request the number of cpu per task
#SBATCH --cpus-per-task=5
# This asks for 3 days
#SBATCH -t 3-00:00:00
# Specify memory per core
#SBATCH --mem-per-cpu=9000M
# Insert your own username to get e-mail notifications
#SBATCH --mail-user=ejohn16@liverpool.ac.uk
# Notify user by email when certain event types occur
#SBATCH --mail-type=ALL

# For carrying out MPI gmx runs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge
module load apps/gromacs/2019.3/gcc-5.5.0+openmpi-1.10.7+fftw3_float-3.3.4+fftw3_double-3.3.4+atlas-3.10.3

# List all modules
module list
#
#
echo =========================================================
echo SLURM job: submitted  date = `date`
date_start=`date +%s`

hostname
echo Current directory: `pwd`

echo "Print the following environmetal variables:"
echo "Job name                     : $SLURM_JOB_NAME"
echo "Job ID                       : $SLURM_JOB_ID"
echo "Job user                     : $SLURM_JOB_USER"
echo "Job array index              : $SLURM_ARRAY_TASK_ID"
echo "Submit directory             : $SLURM_SUBMIT_DIR"
echo "Temporary directory          : $TMPDIR"
echo "Submit host                  : $SLURM_SUBMIT_HOST"
echo "Queue/Partition name         : $SLURM_JOB_PARTITION"
echo "Node list                    : $SLURM_JOB_NODELIST"
echo "Hostname of 1st node         : $HOSTNAME"
echo "Number of nodes allocated    : $SLURM_JOB_NUM_NODES or $SLURM_NNODES"
echo "Number of tasks              : $SLURM_NTASKS"
echo "Number of tasks per node     : $SLURM_TASKS_PER_NODE"
echo "Initiated tasks per node     : $SLURM_NTASKS_PER_NODE"
echo "Requested CPUs per task      : $SLURM_CPUS_PER_TASK"
echo "Requested CPUs on the node   : $SLURM_CPUS_ON_NODE"
echo "Scheduling priority          : $SLURM_PRIO_PROCESS"

#############################################

##Script to set up trimer equilibrium replicates in a batch for RAMD analysis

echo "Running parallel job:"
echo Job output begins
echo -----------------

#Assign variable to indicate where files are located
DATA_IN=/mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/RAMD/trimers

for i in homotrimer heterotrimer apo-homotrimer apo-heterotrimer; do
    for j in 1 2 3 4 5; do
    
    DIR=${DATA_IN}/${i}/GROMACS/Replica${j}
    mkdir -p ${DIR}
    cd ${DIR}


    #Create topology, piping in '6' for Amber99sb-ildn force field
    echo "Creating topology for the ${i} monomer, replicate ${j}, using Amber99sb-ildn force field and virtual hydrogens..."
    printf "6\n" | gmx pdb2gmx -f ${DATA_IN}/pdb_files/${i}_0ns_z.pdb -o ${DIR}/${i}_R${j}.gro -water tip3p
    echo "Topologies created!"
    echo -----------------

    #Define and solvate box
    gmx editconf -f ${DIR}/${i}_R${j}.gro -o ${DIR}/${i}_R${j}_newbox.gro -c -d 1.5
    gmx solvate -cp ${DIR}/${i}_R${j}_newbox.gro -cs spc216.gro -o ${DIR}/${i}_R${j}_solv.gro -p ${DIR}/topol.top

    #Add ions
    echo "Neutralising system..."
    gmx grompp -f ${DATA_IN}/mdp_files/ions.mdp -c ${DIR}/${i}_R${j}_solv.gro -p ${DIR}/topol.top -o ${DIR}/${i}_R${j}_ions.tpr -maxwarn 1
    #Piping in 'SOL' so solvent is replaced with ions
    printf "SOL\n" | gmx genion -s ${DIR}/${i}_R${j}_ions.tpr -o ${DIR}/${i}_R${j}_ions.gro -p ${DIR}/topol.top -pname K -nname CL -neutral -conc 0.15
    echo "System neutralised!"
    echo -----------------

    #Carry out energy minimisation
    echo "Assembling binary for energy minimisation..."
    gmx grompp -f ${DATA_IN}/mdp_files/minim.mdp -c ${DIR}/${i}_R${j}_ions.gro -p ${DIR}/topol.top -o ${DIR}/${i}_R${j}_em.tpr
    #Invoke mdrun to carry out energy minimisation
    echo "Carrying out energy minimisation..."
    mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_em.tpr -deffnm ${i}_R${j}_em
    echo "Minimisation complete!"
    echo -----------------

    #Generate chart of energy minimisation
    printf "Potential\nzero\n" | gmx energy -f ${DIR}/${i}_R${j}_em.edr -o ${DIR}/${i}_R${j}_potential.xvg
    echo "Energy minimisation plotted!"
    echo -----------------


    #Carry out  ensemble
    echo "Assembling binary for NVT, ${i} monomer, replicate ${j} at ${TEMP[${j}]} kelvin..."
    gmx grompp -f ${DATA_IN}/mdp_files/nvt.mdp -c ${DIR}/${i}_R${j}_em.gro -r ${DIR}/${i}_R${j}_em.gro -p ${DIR}/topol.top -o ${DIR}/${i}_R${j}_nvt.tpr
    #Invoke mdrun to carry out NVT ensemble equilibration
    echo "Invoking mdrun to carry out NVT at ${TEMP[${j}]} kelvin..."
    mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_nvt.tpr -deffnm ${i}_R${j}_nvt
    echo "Temperature equilibration complete!"
    echo -----------------

    #Generate chart of temperature
    printf "Temperature\nzero\n" | gmx energy -f ${DIR}/${i}_R${j}_nvt.edr -o ${DIR}/${i}_R${j}_temperature.xvg
    echo "Temperature equilibration for ${i} monomer, replicate ${j}, plotted! Temperature should be ${TEMP[${j}]} kelvin."
    echo -----------------

    #Carry out NPT ensemble
    echo "Assembling binary for NPT, ${i} monomer, replicate ${j} at ${TEMP[${j}]} kelvin..."
    gmx grompp -f ${DATA_IN}/mdp_files/npt.mdp -c ${DIR}/${i}_R${j}_nvt.gro -r ${DIR}/${i}_R${j}_nvt.gro -p ${DIR}/topol.top -o ${DIR}/${i}_R${j}_npt.tpr
    #Invoke mdrun to carry out NPT ensemble equilibration
    echo "Invoking mdrun to carry out NPT..."
    mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_npt.tpr -deffnm ${i}_R${j}_npt
    echo "Pressure and density equilibration complete!"
    echo -----------------

    #Generate charts of pressure and density
    printf "Pressure\nzero\n" | gmx energy -f ${DIR}/${i}_R${j}_npt.edr -o ${DIR}/${i}_R${j}_pressure.xvg
    echo "Pressure equilibration for ${i} plotted!"
    echo -----------------
    printf "Density\nzero\n" | gmx energy -f ${DIR}/${i}_R${j}_npt.edr -o ${DIR}/${i}_R${j}_density.xvg
    echo "Density equilibration for ${i} plotted!"
    echo -----------------

    #MD
    #Assemble binary with grompp
    echo "Running gromacs preprocessor(grompp) to make a run input file..."
    gmx grompp -f ${DATA_IN}/mdp_files/md_0.mdp -c ${DIR}/${i}_R${j}_npt.gro -r ${DIR}/${i}_R${j}_npt.gro -p ${DIR}/topol.top -o ${DIR}/${i}_R${j}_md_1.tpr -maxwarn 5
  
    echo
    echo -----------------
  
    #Begin production MD
    echo "Running main gromacs engine (mdrun) to perform a simulation, do a normal mode analysis or an energy minimization."
    echo "Running command: gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s STRUCTURE.tpr -deffnm JOB_NAME"
    mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_md_1.tpr -deffnm ${i}_R${j}_md_1
    echo "Simulation complete!"

    done

done


############################################

ret=$?

echo
echo ---------------
echo Job output ends
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))
echo =========================================================
echo SLURM job: finished   date = `date`
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================
exit $ret
