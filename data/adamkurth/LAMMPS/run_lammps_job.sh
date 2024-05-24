#!/bin/bash

# Load LAMMPS module
echo "Loading LAMMPS: lammps/29Sep2021 --------------------------"
module load lammps/29Sep2021

# Change directory to LAMMPS project
echo "Changing to LAMMPS project directory ------------------------------"
cd /home/amkurth/Development/lammps_project

# Check for correct number of arguments
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 NAME INPUT_SCRIPT TASKS PARTITION QOS TIME TAG"
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
RUN="${NAME}_${TAG}"  # Use underscore for clarity
SLURMFILE="${RUN}.sh"
OUTPUT_DIR="./${RUN}"

# Print input arguments
echo "Parameters: NAME=$NAME, INPUT_SCRIPT=$INPUT_SCRIPT, TASKS=$TASKS, PARTITION=$PARTITION, QOS=$QOS, TIME=$TIME"
echo "Submitting to SLURM ------------------------------"

# Create SLURM job script
echo "#!/bin/sh" > $SLURMFILE
echo "#SBATCH --job-name=$RUN" >> $SLURMFILE
echo "#SBATCH --output=$OUTPUT_DIR/%x.out" >> $SLURMFILE
echo "#SBATCH --error=$OUTPUT_DIR/%x.err" >> $SLURMFILE
echo "#SBATCH --time=$TIME" >> $SLURMFILE
echo "#SBATCH --ntasks=$TASKS" >> $SLURMFILE
echo "#SBATCH --partition=$PARTITION" >> $SLURMFILE
echo "#SBATCH --qos=$QOS" >> $SLURMFILE
echo "#SBATCH --chdir=$PWD" >> $SLURMFILE
echo "" >> $SLURMFILE

# Specify LAMMPS executable
LAMMPS_EXEC="lmp"

# Add command to run LAMMPS
echo "$LAMMPS_EXEC -in $INPUT_SCRIPT > $OUTPUT_DIR/$RUN.out 2> $OUTPUT_DIR/$RUN.err" >> $SLURMFILE

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Submit job
sbatch $SLURMFILE

echo "Submitted SLURM job: $RUN ------------------------------"

