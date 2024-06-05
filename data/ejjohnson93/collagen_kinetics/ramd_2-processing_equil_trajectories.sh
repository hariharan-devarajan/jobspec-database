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
#SBATCH -N 1
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

##Script to process outputs of equilibrium simulations for pulling and RAMD

echo "Running parallel job:"
echo Job output begins
echo -----------------

#Assign variable to indicate where files are located
DATA_IN=/mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/RAMD/trimers

# Holo- trimers and apo- trimers must be processed seperately due to calcium
# Would use ifelse statement to control this in future but in this instance just used two seperate loops

for i in homotrimer heterotrimer; do
    for j in 1 2 3 4 5; do
    
    DIR=${DATA_IN}/${i}/GROMACS/Replica${j}
    mkdir -p ${DIR}
    cd ${DIR}

    ####Create index

    printf "Protein\nSystem\n" | gmx trjconv -pbc none -center -s ${i}_R${j}_nvt.tpr -f ${i}_R${j}_md_1.xtc -dump 0 -o ${i}_R${j}_frame1_0ns_whole_system.pdb

    # indexes for individual chains
    for k in A B C; do

        gmx select -s ${i}_R${j}_frame1_0ns_whole_system.pdb -on chain_${k}.ndx -select "chain ${k}"

    done

    # indexes for interface and protein + ions
    gmx select -s ${i}_R${j}_frame1_0ns_whole_system.pdb -on interface.ndx -select "chain A and resid 135"
    gmx select -s ${i}_R${j}_frame1_0ns_whole_system.pdb -on ${i}_ions.ndx -select "ion"
    gmx select -s ${i}_R${j}_frame1_0ns_whole_system.pdb -on ${i}_calcium.ndx -select "CA"
    gmx select -s ${i}_R${j}_frame1_0ns_whole_system.pdb -on ${i}_protein_and_ion.ndx -select "protein"
    printf "q\n" | gmx make_ndx -f ${i}_R${j}_nvt.tpr -o general_index.ndx

    sed -i 's/Protein/Protein_ions/g' ${i}_protein_and_ion.ndx
    tail -n +2 ${i}_calcium.ndx | cat >>${i}_protein_and_ion.ndx

    # join the indexes together
    cat general_index.ndx chain_A.ndx chain_B.ndx chain_C.ndx interface.ndx ${i}_protein_and_ion.ndx > ${i}_index.ndx

    # test if needed 
    # printf "Protein\nProtein_ions\n" | gmx trjconv -pbc none -center -s ${i}_R${j}_nvt.tpr -f ${i}_R${j}_md_1.xtc -dump 0 -o test.pdb -n ${i}_index.ndx


    ####Trjconv - remove PBC
    
    echo "Removing PBC..."
    printf "chain_A_and_resid_135\nSystem\n" | gmx trjconv -f ${i}_R${j}_md_1.xtc -s ${i}_R${j}_nvt.tpr -o ${i}_R${j}_trj_combined_nojump.xtc -center -pbc nojump -n ${i}_index.ndx
    #'chain_A_and_resid_135' for centering
    #'System' for output
  
    printf "chain_A_and_resid_135\nSystem\n" | gmx trjconv -f ${i}_R${j}_trj_combined_nojump.xtc -s ${i}_R${j}_nvt.tpr -o ${i}_R${j}_trj_combined_removepbc.xtc -center -pbc mol -ur compact -n ${i}_index.ndx
    #'chain_A_and_resid_135'
    #'System'
  
    ####Trjconv - remove rotation
    
    printf "Protein_ions\nchain_A_and_resid_135\nProtein_ions\n" | gmx trjconv -f ${i}_R${j}_trj_combined_removepbc.xtc -s ${i}_R${j}_nvt.tpr -o ${i}_R${j}_trj_no_rotation_protein.xtc -center -fit rot+trans -n ${i}_index.ndx
    #'Protein'
    #'chain_A_and_resid_135'
    #'Protein'
  
    printf "Protein\nchain_A_and_resid_135\nSystem\n" | gmx trjconv -f ${i}_R${j}_trj_combined_removepbc.xtc -s ${i}_R${j}_nvt.tpr -o ${i}_R${j}_trj_no_rotation.xtc -center -fit rot+trans -n ${i}_index.ndx
    #'Protein'
    #'chain_A_and_resid_135'
    #'System'
  
    ####Trjconv - get starting structure and 10 ns structure
  
    # 0 ns
    printf "chain_A_and_resid_135\nProtein_ions\n" | gmx trjconv -f ${i}_R${j}_trj_no_rotation.xtc -s ${i}_R${j}_nvt.tpr -dump 0 -o ${i}_R${j}_protein_time_0.pdb -center -n ${i}_index.ndx

    # 10 ns
    printf "chain_A_and_resid_135\nProtein_ions\n" | gmx trjconv -f ${i}_R${j}_trj_no_rotation.xtc -s ${i}_R${j}_nvt.tpr -dump 10000 -o ${i}_R${j}_protein_time_10.pdb -center -n ${i}_index.ndx


    done

done


for i in apo-homotrimer apo-heterotrimer; do
    for j in {1..10}; do
    
    DIR=${DATA_IN}/${i}/GROMACS/Replica${j}
    mkdir -p ${DIR}
    cd ${DIR}

    ####Create index

    printf "Protein\nSystem\n" | gmx trjconv -pbc none -center -s ${i}_R${j}_nvt.tpr -f ${i}_R${j}_md_1.xtc -dump 0 -o ${i}_R${j}_frame1_0ns_whole_system.pdb

    # indexes for individual chains
    for k in A B C; do

        gmx select -s ${i}_R${j}_frame1_0ns_whole_system.pdb -on chain_${k}.ndx -select "chain ${k}"

    done

    # indexes for interface and protein + ions
    gmx select -s ${i}_R${j}_frame1_0ns_whole_system.pdb -on interface.ndx -select "chain A and resid 135"
    printf "q\n" | gmx make_ndx -f ${i}_R${j}_nvt.tpr -o general_index.ndx


    # join the indexes together
    cat general_index.ndx chain_A.ndx chain_B.ndx chain_C.ndx interface.ndx > ${i}_index.ndx

    # test if needed 
    # printf "Protein\nProtein_ions\n" | gmx trjconv -pbc none -center -s ${i}_R${j}_nvt.tpr -f ${i}_R${j}_md_1.xtc -dump 0 -o test.pdb -n ${i}_index.ndx


    ####Trjconv - remove PBC
    
    echo "Removing PBC..."
    printf "chain_A_and_resid_135\nSystem\n" | gmx trjconv -f ${i}_R${j}_md_1.xtc -s ${i}_R${j}_nvt.tpr -o ${i}_R${j}_trj_combined_nojump.xtc -center -pbc nojump -n ${i}_index.ndx
    #'chain_A_and_resid_135' for centering
    #'System' for output
  
    printf "chain_A_and_resid_135\nSystem\n" | gmx trjconv -f ${i}_R${j}_trj_combined_nojump.xtc -s ${i}_R${j}_nvt.tpr -o ${i}_R${j}_trj_combined_removepbc.xtc -center -pbc mol -ur compact -n ${i}_index.ndx
    #'chain_A_and_resid_135'
    #'System'
  
    ####Trjconv - remove rotation
    
    printf "Protein_ions\nchain_A_and_resid_135\nProtein_ions\n" | gmx trjconv -f ${i}_R${j}_trj_combined_removepbc.xtc -s ${i}_R${j}_nvt.tpr -o ${i}_R${j}_trj_no_rotation_protein.xtc -center -fit rot+trans -n ${i}_index.ndx
    #'Protein'
    #'chain_A_and_resid_135'
    #'Protein'
  
    printf "Protein\nchain_A_and_resid_135\nSystem\n" | gmx trjconv -f ${i}_R${j}_trj_combined_removepbc.xtc -s ${i}_R${j}_nvt.tpr -o ${i}_R${j}_trj_no_rotation.xtc -center -fit rot+trans -n ${i}_index.ndx
    #'Protein'
    #'chain_A_and_resid_135'
    #'System'
  
    ####Trjconv - get starting structure and 10 ns structure
  
    # 0 ns
    printf "chain_A_and_resid_135\nProtein\n" | gmx trjconv -f ${i}_R${j}_trj_no_rotation.xtc -s ${i}_R${j}_nvt.tpr -dump 0 -o ${i}_R${j}_protein_time_0.pdb -center -n ${i}_index.ndx

    # 10 ns
    printf "chain_A_and_resid_135\nProtein\n" | gmx trjconv -f ${i}_R${j}_trj_no_rotation.xtc -s ${i}_R${j}_nvt.tpr -dump 10000 -o ${i}_R${j}_protein_time_10.pdb -center -n ${i}_index.ndx


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
