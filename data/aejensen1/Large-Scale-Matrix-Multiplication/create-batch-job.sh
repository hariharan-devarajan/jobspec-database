#!/bin/bash

# Check if input is a number. If not, exit the script.
validate_number() {
    if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ]; then
        echo "Error: Input is not a valid positive number. Exiting the script."
        exit 1
    fi
}

# Parse command-line arguments
# Check if the number of arguments is equal to 6, otherwise print usage instruction
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <file_type> <array_size> <num_threads> <job_name> <num_nodes> <ntasks-per-node>"
    echo "Available file types: cNoThread, cPThread, cOpenMP, pythonNoThreads, pythonWithThreads"
    echo "Max number of threads: 256 (128 cores * 2 threads per core)"
    exit 1
fi

file_type=$1
array_size=$2
num_threads=$3 # Max = 256
job_name=$4
num_nodes=$5
ntasks_per_node=$6

# Validate file type
case "$file_type" in
    "cNoThread" | "cPThread" | "cOpenMP" | "pythonNoThreads" | "pythonWithThreads" | "cNoThread.c" | "cPThread.c" | "cOpenMP.c" | "pythonNoThreads.py" | "pythonWithThreads.py") ;;
    *)
	echo "Error: Unsupported file type: $file_type. Exiting."
	exit 1
	;;
esac

# Validate array size and number of threads
validate_number "$array_size"
validate_number "$num_threads"
validate_number "$num_nodes"
validate_number "$ntasks_per_node"

# Check that the number of threads is less than or equal to 256.
if [ "$num_threads" -gt 256 ]; then
    echo "Error: The number of threads cannot exceed 256. Exiting."
    exit 1
fi

# Calculate the number of cores needed
num_cores=$((num_threads + 1 / 2)) # Hyperthreading (2 threads per core)

program_type=""
case "$file_type" in
    "cNoThread" | "cNoThread.c")
        program_type="${job_name}_cNoThread.c"
        if [ -f "./compiled-programs/$program_type" ]; then
            rm "./compiled-programs/$program_type"
        fi
        gcc -o "./compiled-programs/$program_type" cNoThread.c ;;
    "cPThread" | "cPThread.c")
        program_type="${job_name}_cPThread.c"
        if [ -f "./compiled-programs/$program_type" ]; then
            rm "./compiled-programs/$program_type"
        fi
        gcc -o "./compiled-programs/$program_type" cPThread.c -lpthread ;;
    "cOpenMP" | "cOpenMP.c")
        program_type="${job_name}_cOpenMP.c"
        if [ -f "./compiled-programs/$program_type" ]; then
            rm "./compiled-programs/$program_type"
        fi
        gcc -o "./compiled-programs/$program_type" cOpenMP.c -fopenmp ;;
    "pythonNoThreads" | "pythonNoThreads.py")
        program_type="pythonNoThreads.py" ;;
    "pythonWithThreads" | "pythonWithThreads.py")
        program_type="pythonWithThreads.py" ;;
esac


# Generate Slurm batch job script
batch_job_script="job_${job_name}.sbatch"

# Write the Slurm batch job script with either C program or Python script execution
if [[ "$program_type" == *.c ]]; then
    cat <<EOF >"./batch-jobs/$batch_job_script"
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --nodes=$num_nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH --cpus-per-task=$num_cores
#SBATCH --time=4:00:00
#SBATCH -o ./slurm-output/output.%j.out # STDOUT

module load gcc

./compiled-programs/$program_type $array_size $num_threads $num_cores $job_name $num_nodes $ntasks_per_node
EOF
else
    cat <<EOF >"./batch-jobs/$batch_job_script"
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --nodes=$num_nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH --cpus-per-task=$num_cores
#SBATCH --time=4:00:00
#SBATCH -o ./slurm-output/output.%j.out # STDOUT

module load python

python "$program_type" $array_size $num_threads $num_cores $job_name $num_nodes $ntasks_per_node
EOF
fi

# Submit the Slurm batch job
echo "Submitting Slurm batch job with program $file_type and array size $array_size with $num_threads threads.."

sbatch "./batch-jobs/$batch_job_script"

echo "Slurm batch job submitted successfully."
