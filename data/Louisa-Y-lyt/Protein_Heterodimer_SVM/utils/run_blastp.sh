#!/bin/bash
#SBATCH --job-name=blastp
#SBATCH --time=4:00:00
#SBATCH --qos=high
#SBATCH -e "/n/home10/ytingliu/alphapulldown_new/logs/blastp_%A_%a_err.txt"
#SBATCH -o "/n/home10/ytingliu/alphapulldown_new/logs/blastp_%A_%a_out.txt"
#SBATCH --mem=128000
#SBATCH --mail-type=END
#SBATCH --mail-user=yutingliu@hsph.harvard.edu
#SBATCH -N 1
#SBATCH --cpus-per-task=4  # Adjust the number of CPU cores per task as needed

# Define the input query file and database directory
INPUT_FILE=$1
SLURM_CPUS_PER_TASK=4
# Define the blastp command
BLASTP_CMD="/n/home10/ytingliu/ncbi-blast-2.15.0+/bin/blastp"

# Extract the task ID to determine the sequence to process
TASK_ID=$SLURM_ARRAY_TASK_ID

# Extract the sequence from the input file based on the task ID
HEADER=$(sed -n "$((TASK_ID * 4 - 3))p" $INPUT_FILE)
SEQUENCE=$(sed -n "$((TASK_ID * 4 - 2))p" $INPUT_FILE)
# Write the sequence to a temporary file
TMP_QUERY_FILE="/n/home10/ytingliu/blast_tmp/query_${TASK_ID}.fasta"
echo $HEADER > $TMP_QUERY_FILE
echo $SEQUENCE >> $TMP_QUERY_FILE

# Define the database file based on the task ID
DATABASE="human_protein_db"

# Run blastp command for the current array task
$BLASTP_CMD -query $TMP_QUERY_FILE -db $DATABASE -out /n/home10/ytingliu/blast_res/${HEADER#>}.txt -outfmt 6 -num_threads $SLURM_CPUS_PER_TASK

# Clean up temporary files
rm $TMP_QUERY_FILE
