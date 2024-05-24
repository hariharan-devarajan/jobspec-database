#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job.
#SBATCH --export=ALL
# Define job name
#SBATCH -J RAMD
# Define a standard output file. When the job is running, %u will be replaced by user name,
# %N will be replaced by the name of the node that runs the batch script, and %j will be replaced by job id number.
#SBATCH -o RAMD.%u.%N.%j.out
# Define a standard error file
#SBATCH -e RAMD.%u.%N.%j.err
# Request the partition
#SBATCH -p nodes
# Request the number of nodes
#SBATCH -N 8
# Specify the number of tasks per node
#SBATCH --ntasks-per-node=8
# Specify the number of tasks
##SBATCH --ntasks=16 #this option is not set as we have already set --ntasks-per-node
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
#
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module purge
module load apps/gromacs/2020.5_ramd-2.0

# List all modules
module list
#
#
echo =========================================================
echo SLURM job: submitted date = $(date)
date_start=$(date +%s)

hostname
echo Current directory: $(pwd)

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

##Script to run RAMD simulations for homotrimer

echo "Running parallel job:"
echo Job output begins
echo -----------------

DATA_IN=/mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/RAMD/trimers
MDP=/mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/RAMD/mdp_files_trimers

for i in homotrimer; do
    for j in {1..5}; do

        DIR=${DATA_IN}/${i}/RAMD/Replica${j}_RAMD
        mkdir -p ${DIR}
        cd ${DIR}

        ###########################

        # pdb2gmx

        printf "6\n" | gmx pdb2gmx -f ${DATA_IN}/${i}/GROMACS/Replica${j}/${i}_R${j}_protein_time_10.pdb -o ${i}_R${j}.gro -water tip3p -ignh

        # system set-up
        gmx editconf -f ${i}_R${j}.gro -o ${i}_R${j}_box.gro -c -d 7.5

        gmx solvate -cp ${i}_R${j}_box.gro -cs spc216.gro -o ${i}_R${j}_solvated.gro -p topol.top
        gmx grompp -f /mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/pulling_simulations/mdp_files/ions.mdp -c ${i}_R${j}_solvated.gro -p topol.top -o ${i}_R${j}_ions.tpr
        printf "SOL\n" | gmx genion -s ${i}_R${j}_ions.tpr -o ${i}_R${j}_ions.gro -p topol.top -pname K -nname CL -neutral -conc 0.15

        ####################################

        # minimisation

        gmx grompp -f ${MDP}/minim.mdp -c ${DIR}/${i}_R${j}_ions.gro -p topol.top -o ${DIR}/${i}_R${j}_em.tpr
        mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_em.tpr -deffnm ${i}_R${j}_em

        ############################

        # create chain indexes

        # all-atom pdb file (hydrogens added during conversion to .gro file)
        printf "System\n" | gmx trjconv -f ${DIR}/${i}_R${j}_em.trr -s ${DIR}/${i}_R${j}_em.tpr -o ${DIR}/${i}_R${j}_all_atom.pdb -dump 0

        # extract chains

        # don't need to extract ion chains for partial restraints

        for k in A B C; do

            gmx select -s ${i}_R${j}_all_atom.pdb -on chain_${k}.ndx -select "chain ${k}"

        done

        # name and format chains

        gmx select -s ${i}_R${j}_all_atom.pdb -on alpha1.ndx -select "chain A"
        gmx select -s ${i}_R${j}_all_atom.pdb -on alpha2.ndx -select "chain B"

        # chains that are going to have partial restraints extracted and combined into one index and named 'alpha_1'
        sed -i 's/chain_A/alpha_1/g' alpha1.ndx
        tail -n +2 chain_C.ndx | cat >>alpha1.ndx

        # chain that is going to be pulled along z-axis named 'alpha_2' (regardless of whether simulating a homotrimer where it would actually be an alpha_1 chain, its just so the pull and stationary group can have unique names)
        sed -i 's/chain_B/alpha_2/g' alpha2.ndx

        # create general index and combine with chain indexes
        printf "q\n" | gmx make_ndx -f ${i}_R${j}_ions.gro -o general_index.ndx
        cat general_index.ndx alpha1.ndx alpha2.ndx >${i}_index.ndx

        ############################

        # generate partial restraints
        # only chains A and C are going to have partial restraints as they're the stationary chains

        # create index for group being restrained, append to index file and generate restraint file
        # then append restraint file to the chain topology

        for k in A C; do

            gmx select -s ${i}_R${j}_all_atom.pdb -on ${i}_chain_${k}_backbone.ndx -select "chain ${k} and group "Backbone""
            cat ${i}_chain_${k}_backbone.ndx >>${i}_index.ndx

        done

        # Chain A

        printf "chain_A_and_group_Backbone\n" | gmx genrestr -f ${i}_R${j}_em.tpr -n ${i}_index.ndx -o posre_Backbone_chain_A.itp -fc 1000 1000 1000

        sed 's/FILE_NAME/'posre_Backbone_chain_A.itp'/g' /mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/pulling_simulations/pos_res_conditions/pos_res_template.txt >>topol_Protein_chain_A.itp

        # Chain C
        # Requires additional formatting so numbering is chain specific rather than whole protein specific 

        printf "chain_C_and_group_Backbone\n" | gmx genrestr -f ${i}_R${j}_em.tpr -n ${i}_index.ndx -o TEMP_chainC.itp -fc 1000 1000 1000

        echo | head -n +4 TEMP_chainC.itp >TEMP_header

        tail -n +5 TEMP_chainC.itp | awk '{
       if(NR == 1) {
           shift = ($1 - 1)
       }

       print ($1 - shift) "    " $2 "       " $3 "       " $4 "       " $5
      }' >TEMP_atom_index

        cat TEMP_header TEMP_atom_index >posre_Backbone_chain_C.itp
        rm TEMP_*

        sed 's/FILE_NAME/'posre_Backbone_chain_C.itp'/g' /mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/pulling_simulations/pos_res_conditions/pos_res_template.txt >>topol_Protein_chain_C.itp

        #######################################

        # equilibration and md - run RAMD protocol

        #generate mdp files

        for NAME in {1..25}; do

            sed "s/XX/${NAME}/" ${MDP}/gromacs_ramd_homotrimer.mdp > gromacs_ramd_homotrimer_${NAME}.mdp

            #equilibration
            gmx grompp -f ${MDP}/nvt.mdp -c ${DIR}/${i}_R${j}_em.gro -p topol.top -r ${DIR}/${i}_R${j}_em.gro -o ${DIR}/${i}_R${j}_RAMD-${NAME}_nvt.tpr
            mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_RAMD-${NAME}_nvt.tpr -deffnm ${i}_R${j}_RAMD-${NAME}_nvt

            gmx grompp -f ${MDP}/npt.mdp -c ${DIR}/${i}_R${j}_RAMD-${NAME}_nvt.gro -p topol.top -r ${DIR}/${i}_R${j}_RAMD-${NAME}_nvt.gro -o ${DIR}/${i}_R${j}_RAMD-${NAME}_npt.tpr
            mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_RAMD-${NAME}_npt.tpr -deffnm ${i}_R${j}_RAMD-${NAME}_npt

            # RAMD

            gmx grompp -f gromacs_ramd_homotrimer_${NAME}.mdp -c ${i}_R${j}_RAMD-${NAME}_npt.gro -p topol.top -r ${i}_R${j}_RAMD-${NAME}_npt.gro -o gromacs_ramd-${NAME}.tpr -maxwarn 2 -n ${i}_index.ndx
            mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s gromacs_ramd-${NAME}.tpr -deffnm gromacs_ramd-${NAME} -pf gromacs_ramd-${NAME}_pullf.xvg -px gromacs_ramd-${NAME}_pullx.xvg -maxh 8 > out-${NAME}
            
            rm *.trr

        done

    done

done

############################################

ret=$?

echo
echo ---------------
echo Job output ends
date_end=$(date +%s)
seconds=$((date_end - date_start))
minutes=$((seconds / 60))
seconds=$((seconds - 60 * minutes))
hours=$((minutes / 60))
minutes=$((minutes - 60 * hours))
echo =========================================================
echo SLURM job: finished date = $(date)
echo Total run time : $hours Hours $minutes Minutes $seconds Seconds
echo =========================================================
exit $ret
