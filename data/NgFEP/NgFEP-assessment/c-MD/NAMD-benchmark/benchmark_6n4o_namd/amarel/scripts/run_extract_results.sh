#!/bin/bash
#SBATCH --partition=gpu              #
#SBATCH --gres=gpu:1                 #
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=test            # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                   # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32G                    # Real memory (RAM) required per node
#SBATCH --time=01:00:00              # Total run time limit (DD-HH:MM:SS)
#SBATCH --nodelist=gpu[015-016,019-026]

module load apptainer/1.2.5
#apptainer exec --nv namd3.simg namd3 namd3_input.in > $outfile

device="A100" 

for index in {1..5}; do
outfile="gpu_cuda_${device}_${index}.out"
apptainer exec --nv namd3.simg namd3 namd3_input.in > $outfile
done

cat /proc/cpuinfo | grep "model name" | uniq
grep  TIMING: gpu_cuda_0_*.out | awk '{printf "Performance %f %s ",\$9,\$10}'
echo -n $1 "GPUs, " 
hostname -s

# Output file
output_file="${device}_results_namd.dat"
temp_file="tmpl"

# Clear the output and temporary files if they already exist
rm "$output_file"

# Loop through files matching the pattern
for file in gpu_cuda_${device}_*.out; do
    if [[ -f "$file" ]]; then
        # Calculate the line number for the last 10th line
        total_lines=$(wc -l < "$file")
        line_number=$((total_lines - 23))

        # Extracting the last 10th line from each file
        last_line=$(sed -n "${line_number}p" "$file")

        # Writing the results to the temporary file
        echo "$last_line" >> "$temp_file"
#        echo "" >> "$temp_file"
    fi
done

# Extract ns/day values and write them to the output file
grep "TIMING: " tmpl | awk '{print $9}' > "$output_file"

## Optional: Remove temporary file
rm "$temp_file"

echo "Extracted data saved to $output_file"
