#!/bin/bash

# SBATCH directives
#SBATCH --partition=lva
#SBATCH --job-name benchmark
#SBATCH --output benchmark.log
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# Path to the allscale_api project directory
ALLSCALE_API_DIR=/home/cb76/cb761222/allscale_api/code

# Path to MiMalloc and RPMalloc libraries
MIMALLOC=/home/cb76/cb761222/mimalloc/build/libmimalloc.so
RPMALLOC=/home/cb76/cb761222/rpmalloc/bin/linux/release/x86-64/librpmalloc.so
module load llvm/15.0.4-python-3.10.8-gcc-8.5.0-bq44zh7
# Function to perform benchmark
perform_benchmark() {
    local allocator_lib=$1
    local benchmark_name=$2
    
    echo "Starting benchmark for $benchmark_name with $allocator_lib"

    # Clean previous build artifacts
    cd $ALLSCALE_API_DIR
    ninja clean

    # Set LD_PRELOAD environment variable to preload allocator library
    export LD_PRELOAD=$allocator_lib


    # Extract and log the performance metrics
     /usr/bin/time -f "Real Time (s): %e\nUser CPU Time (s): %U\nSystem CPU Time (s): %S\nPeak Memory (KB): %M" ninja

    # Unset LD_PRELOAD for subsequent benchmarks
    unset LD_PRELOAD
}

# Perform benchmarks
perform_benchmark "" "No Allocator Preloading"           # Benchmark without preloading any allocator
perform_benchmark "$RPMALLOC" "RPMalloc Allocator Preloading"   # Benchmark with RPMalloc preloaded
perform_benchmark "$MIMALLOC" "MiMalloc Allocator Preloading"   # Benchmark with MiMalloc preloaded
