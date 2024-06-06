#!/bin/bash

# SBATCH directives
#SBATCH --partition=lva
#SBATCH --job-name benchmark
#SBATCH --output benchmark.log
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# Delete the existing CSV file
rm -f results.csv

# Define the parameters for the benchmark
data_structures=("Array" "LinkedList")
ins_del_ratios=(0 1 10 50)
read_write_ratios=(100 99 90 50)
element_sizes=(8 512 8192)
num_elements=(10 1000 100000 10000000)

gcc -O3 -o Array array_bench.c
gcc -O3 -o LinkedList linked_list_bench.c

# Initialize a counter for the test sequence
test_sequence=1

# Write the header for the CSV file
echo "idx, data structure, Ins/Del Ratio,Read/Write Ratio,Element Size,Number of Elements,Operations Completed, Time[Seconds]" >> results.csv

# Loop over each combination of parameters 1,100,8,10000000,
for ds in ${data_structures[@]}; do
    for idr in ${ins_del_ratios[@]}; do
        for rwr in ${read_write_ratios[@]}; do
            for es in ${element_sizes[@]}; do
                for ne in ${num_elements[@]}; do
                    # Print the arguments
                    echo "Arguments: $ds $idr $rwr $es $ne"
                    # Run the benchmark and save the output to a file
                    echo -n "$test_sequence,$ds,$idr,$rwr,$es,$ne," >> results.csv
                    ./$ds $idr $rwr $es $ne >> results.csv
                    ((test_sequence++))
                done
            done
        done
    done
done