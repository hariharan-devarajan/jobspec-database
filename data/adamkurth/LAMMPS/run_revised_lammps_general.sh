ng test cases of lammps on agave. 
# This script is a template for submitting LAMMPS simulations to a SLURM queueing system. 

# load lammps                                                                   
echo "Loading LAMMPS: lammps/29Sep2021 --------------------------"               
module load lammps/29Sep2021                                                    

if [ $? -ne 0 ]; then
    echo "Failed to load LAMMPS module. Exiting."
    exit 1
fi

#change directory                                                               
echo "changing to LAMMPS directory ------------------------------"               
cd /home/amkurth/Development/lammps_project                      

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 NAME INPUT_SCRIPT TASKS PARTITION QOS TIME"
    exit 1
fi

# Extract arguments
NAME="$1"
INPUT_SCRIPT="$2"
TASKS="$3"
PARTITION="$4"
QOS="$5"
TIME="$6"

# Define simulation parameters
RUN="sim_${NAME}"  # TAG removed
SLURMFILE="${RUN}.sh"
OUTPUT_DIR="./${RUN}"

# Print the input arguments
echo "NAME: $NAME, INPUT_SCRIPT: $INPUT_SCRIPT, TASKS: $TASKS, PARTITION: $PARTITION, 
QOS: $QOS, TIME: $TIME"
echo "Submitting to SLURM ------------------------------"                           

# Create output directory
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory $OUTPUT_DIR already exists. Exiting."
    exit 1
else
    mkdir -p "$OUTPUT_DIR"
fi

# Create the SLURM job script
echo "#!/bin/sh" > $SLURMFILE
echo "#SBATCH --job-name=$RUN" >> $SLURMFILE
echo "#SBATCH --output=$OUTPUT_DIR/%x.out" >> $SLURMFILE
echo "#SBATCH --error=$OUTPUT_DIR/%x.err" >> $SLURMFILE
echo "#SBATCH --time=$TIME" >> $SLURMFILE
echo "#SBATCH --ntasks=$TASKS" >> $SLURMFILE
echo "#SBATCH --partition=$PARTITION" >> $SLURMFILE
echo "#SBATCH --qos=$QOS" >> $SLURMFILE  # Add QOS to SLURM script
echo "#SBATCH --chdir=$PWD" >> $SLURMFILE
echo "" >> $SLURMFILE

# Ensure the LAMMPS executable path is correct for your environment
LAMMPS_EXEC="lmp"  # Adjust for your LAMMPS installation, e.g., lmp_mpi for MPI

# Add the command to run LAMMPS

dd the command to run LAMMPS
echo "cd $OUTPUT_DIR" >> $SLURMFILE
echo "$LAMMPS_EXEC -in $INPUT_SCRIPT -log $OUTPUT_DIR/log.lammps > $RUN.out 2> $RUN.err" >> $SLURMFILE

# Move the SLURM file to the output directory
mv $SLURMFILE $OUTPUT_DIR

# Submit the job
sbatch $OUTPUT_DIR/$SLURMFILE

echo "Submitted SLURM job: $RUN ------------------------------"
