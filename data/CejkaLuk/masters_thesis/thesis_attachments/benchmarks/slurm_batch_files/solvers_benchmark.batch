#!/bin/sh
#SBATCH --partition=amdgpulong
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --out=solvers_benchmark.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --exclude=g[11-12]
#SBATCH --mail-user=cejkaluk@fjfi.cvut.cz
#SBATCH --mail-type=ALL

# Modules
module load CMake/3.24.3-GCCcore-12.2.0
module load CUDA/12.0.0
module load GCC/12.2.0
ml

# Information about the machine
/bin/hostname
/bin/pwd
nvidia-smi

PERSONAL="/mnt/personal/cejkaluk"
SCRATCH="/data/temporary/cejkaluk"

# Create new directory in temporary (scratch) memory
date_time=$(date '+%Y-%m-%d_%H-%M-%S')
SCRATCH_date="${SCRATCH}/${date_time}"

echo "--> Creating new scratch dir: $SCRATCH_date"
mkdir -p $SCRATCH_date

# Copy TNL files from .local to temporary (scratch) memory
echo "--> Copying TNL include..."
cd $PERSONAL

cp -r .local $SCRATCH_date/

# Copy Decomposition project files to temporary (scratch) memory
echo "--> Copying decomposition..."
cp -r decomposition $SCRATCH_date

# Go to the benchmark
cd $SCRATCH_date/decomposition/src/Benchmarks/Solvers/scripts/

# Get short commit SHA
commit_short_sha=$(git rev-parse --short HEAD)

# Run the benchmark
echo "--> Current directory: $(/bin/pwd)"
echo "--> Current commit: $commit_short_sha"

./run-solvers-benchmark "$SCRATCH_date" |& tee benchmark_log_solvers_${commit_short_sha}.txt

# Copy logs to permanent storage
benchmark_logs_dir="${PERSONAL}/matrices-logs/logs/decomposition"
benchmark_run_dir="${benchmark_logs_dir}/${date_time}_${commit_short_sha}_SOLVERS"

mkdir ${benchmark_run_dir}
cp -r log-files ${benchmark_run_dir}
cp benchmark_log*.txt ${benchmark_run_dir}

# Clean up after the benchmark - remove the created scratch dir
rm -rf $SCRATCH_date
