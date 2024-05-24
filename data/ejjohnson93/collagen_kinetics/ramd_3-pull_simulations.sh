#!/bin/bash -l
# Use the current working directory
#SBATCH -D ./
# Use the current environment for this job.
#SBATCH --export=ALL
# Define job name
#SBATCH -J pull
# Define a standard output file. When the job is running, %u will be replaced by user name,
# %N will be replaced by the name of the node that runs the batch script, and %j will be replaced by job id number.
#SBATCH -o gromacs.%u.%N.%j.out
# Define a standard error file
#SBATCH -e gromacs.%u.%N.%j.err
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
module load apps/gromacs/2019.3/gcc-5.5.0+openmpi-1.10.7+fftw3_float-3.3.4+fftw3_double-3.3.4+atlas-3.10.3

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

##Script to set up and carry out pulling simulations

echo "Running parallel job:"
echo Job output begins
echo -----------------

DATA_IN=/mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/RAMD/trimers
MDP=/mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/pulling_simulations/mdp_files/group_test_mdp

for i in homotrimer heterotrimer apo-heterotrimer apo-homotrimer; do
    for j in {1..5}; do
        
        DIR=${DATA_IN}/${i}/PULL/Replica${j}
        mkdir -p ${DIR}
        cd ${DIR}

        ###########################

        # pdb2gmx

        printf "6\n" | gmx pdb2gmx -f ${DATA_IN}/${i}/GROMACS/Replica${j}/${i}_R${j}_protein_time_10.pdb -o ${i}_R${j}.gro -water tip3p -ignh

        # system set-up

        gmx editconf -f ${i}_R${j}.gro -o ${i}_R${j}_pull_box.gro -center 6 6.5 8 -box 12 13 32
        gmx solvate -cp ${i}_R${j}_pull_box.gro -cs spc216.gro -o ${i}_R${j}_pull_solvated.gro -p topol.top
        gmx grompp -f /mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/pulling_simulations/mdp_files/ions.mdp -c ${i}_R${j}_pull_solvated.gro -p topol.top -o ${i}_R${j}_pull_ions.tpr
        printf "SOL\n" | gmx genion -s ${i}_R${j}_pull_ions.tpr -o ${i}_R${j}_pull_ions.gro -p topol.top -pname K -nname CL -neutral -conc 0.15

        ####################################

        # minimisation

        gmx grompp -f ${MDP}/minim.mdp -c ${DIR}/${i}_R${j}_pull_ions.gro -p topol.top -o ${DIR}/${i}_R${j}_pull_em.tpr
        mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_pull_em.tpr -deffnm ${i}_R${j}_pull_em

        ############################

        # create chain indexes

        # all-atom pdb file (hydrogens added during conversion to .gro file)
        printf "System\n" | gmx trjconv -f ${DIR}/${i}_R${j}_pull_em.trr -s ${DIR}/${i}_R${j}_pull_em.tpr -o ${DIR}/${i}_R${j}_all_atom.pdb -dump 0

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

        # chain that is going to be pulled along z-axis named 'alpha_2' (regardless of whether simulating a homotrimer where it would actually be an alpha_1 chain - this is so the pull and stationary group can have unique names)
        # in future work i would just name the groups 'stationary' and 'mobile' 
        
        sed -i 's/chain_B/alpha_2/g' alpha2.ndx

        # create general index and combine with chain indexes
        printf "q\n" | gmx make_ndx -f ${i}_R${j}_pull_ions.gro -o general_index.ndx
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

        printf "chain_A_and_group_Backbone\n" | gmx genrestr -f ${i}_R${j}_pull_em.tpr -n ${i}_index.ndx -o posre_Backbone_chain_A.itp -fc 1000 1000 1000

        sed 's/FILE_NAME/'posre_Backbone_chain_A.itp'/g' /mnt/lustre/users/ejohn16/molecular_dynamics/typeI_col/pulling_simulations/pos_res_conditions/pos_res_template.txt >>topol_Protein_chain_A.itp

        # Chain C
        # This chain needs a little extra processing as the atom index is for the whole protein rather than chain specific
        # Included awk script to modify it to be chain specific

        printf "chain_C_and_group_Backbone\n" | gmx genrestr -f ${i}_R${j}_pull_em.tpr -n ${i}_index.ndx -o TEMP_chainC.itp -fc 1000 1000 1000

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

        # equilibration and md

        #generate mdp files
        
        sed -i 's/MOBILE/alpha_2/g;s/STATIONARY/alpha_1/g' ${MDP}/md_pull_medium.mdp > ${DIR}/md_pull_medium.mdp

        for k in {1..10}; do

            #equilibration

            gmx grompp -f ${MDP}/npt.mdp -c ${DIR}/${i}_R${j}_pull_em.gro -p topol.top -r ${DIR}/${i}_R${j}_pull_em.gro -o ${DIR}/${i}_R${j}_PULL${k}_npt.tpr
            mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_PULL${k}_npt.tpr -deffnm ${i}_R${j}_PULL${k}_npt

            #md

            # medium pull

            gmx grompp -f ${DIR}/md_pull_medium.mdp -c ${DIR}/${i}_R${j}_PULL${k}_npt.gro -p ${DIR}/topol.top -r ${DIR}/${i}_R${j}_PULL${k}_npt.gro -n ${DIR}/${i}_index.ndx -t ${DIR}/${i}_R${j}_PULL${k}_npt.cpt -o ${DIR}/${i}_R${j}_PULL${k}_medium.tpr
            mpirun gmx_mpi mdrun -ntomp $SLURM_CPUS_PER_TASK -v -s ${DIR}/${i}_R${j}_PULL${k}_medium.tpr -deffnm ${i}_R${j}_PULL${k}_medium -pf ${DIR}/${i}_R${j}_pullf-${k}_medium.xvg -px ${DIR}/${i}_R${j}_pullx-${k}_medium.xvg
            
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
