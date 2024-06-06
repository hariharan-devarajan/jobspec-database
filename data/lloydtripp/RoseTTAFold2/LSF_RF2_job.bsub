#!/bin/bash
# Author : Lloyd Tripp
# Start Date: 2024-2-10
# Script Name: LSF_RF2_batcher.bsub
# Description: This script is used to batch process RF2 files using LSF.
# Scope: This script is intended for use in the LT2_Protein-Modeling project.
# Usage: ./LSF_RF2_batcher.bsub
# Expected Input: 
#   - Manifest file containing RF2 parameters for each job.
# Expected Output: Processed RF2 files stored in the specified output directory.
# BSUB Parameters
#BSUB -e RF2_Job%J_%I.err
#BSUB -o RF2_Job%J_%I.out
#BSUB -q general
#BSUB -G compute-ghaller
#BSUB -n 3
#BSUB -R 'gpuhost rusage[mem=64GB]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB"
#BSUB -a "docker(lloydtripp/rf2:latest)" /bin/bash

# Check the number of provided arguments
#if [ "$#" -lt 1 ]; then
#  echo "Usage: $0 <Manifest file path>"
#  exit 1
#fi

# Assign the script parameters
manifest_file_location=$MANIFEST_LOC
echo $(($LSB_JOBINDEX + 1))
row_number=$(($LSB_JOBINDEX + 1)) # Add 1 to the row number to account for the header row

# Check if the manifest file exists and is a file
if [ ! -f "$manifest_file_location" ]; then
  echo "Manifest file not found: $manifest_file_location"
  exit 1
fi

# Check if row number is valid
total_rows=$(($(wc -l < "$manifest_file_location")+1)) 
if [ "$row_number" -le 0 ] || [ "$row_number" -gt "$total_rows" ]; then
  echo "Invalid row number: $row_number"
  exit 1
fi

# Open the manifest file
# Get the header names from the first line of the TSV file
read -r -a header_names < "$manifest_file_location"

# Function to get the value of a parameter by header name
get_parameter_value() {
    local header="$1"
    local parameter_value=""
    local column_index=1
    
    # Get the column index of the given header in the TSV file
    for ((i=0; i<${#header_names[@]}; i++)); do
      if [[ "${header_names[$i]}" == "$header" ]]; then
        column_index=$((i+1))
        break
      fi
    done

    # Read the parameter value in the given row and column
    parameter_value=$(awk -F'\t' -v row_number="$row_number" -v column_index="$column_index" 'NR==row_number {print $column_index}' "$manifest_file_location")

    # Return the parameter value
    echo "$parameter_value"
}

# Usage example: Getting values for specific headers
sample_id=$(get_parameter_value "sample_id")
fasta_file_location=$(get_parameter_value "fasta_file_location")
output_directory=$(get_parameter_value "output_directory")"/"$sample_id 
additional_parameters=$(get_parameter_value "additional_parameters")

# Print the values
echo "Fasta file location: $fasta_file_location"
echo "Output directory: $output_directory"
echo "RF2 Additional Parameters: $additional_parameters"

# Make output_directory if it does not exist
mkdir -p "$output_directory"

# Activate environment
source activate RF2
# Run RF2
# TO DO: Parameterize the RF2 script location.
/storage1/fs1/ghaller/Active/lloydt/LT2_Protein-Modeling/RosettaFold2/run_RF2.sh $fasta_file_location -o $output_directory $additional_parameters 

