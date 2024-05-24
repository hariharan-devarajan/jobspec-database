#!/bin/bash
#SBATCH --partition=amdgpulong
#SBATCH --time=72:00:00
#SBATCH --mem=1000G
#SBATCH --out=BDDCML_benchmark.out
#SBATCH --nodes=1
#SBATCH --ntasks=32  		    # Number of MPI ranks
#SBATCH --ntasks-per-node=32   # Must be == ntasks / nodes 
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:8
#SBATCH --mail-user=cejkaluk@fjfi.cvut.cz
#SBATCH --mail-type=ALL

# Node management
# Exclude g01-g10 as they only have 4x A100 - we need 8x A100
#SBATCH --exclude=g[01-10]
# DO NOT USE `#SBATCH --exclusive` - it creates issues with new version of Slurm

# Pre-processing (prepare the environemnt)

## Information about the machine
/bin/hostname
/bin/pwd
nvidia-smi

## Directory variables
PERSONAL="/mnt/personal/cejkaluk"
BDDCML="$PERSONAL/BDDCML_AMD"

SCRATCH="/data/temporary/cejkaluk"

## Modules/Libraries
source /home/cejkaluk/bddcml_load_modules_AMD.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BDDCML}/deps/magma/2.7.1/lib
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## Create new directory in temporary (scratch) memory
date_time=$(date '+%Y-%m-%d_%H-%M-%S')
SCRATCH_date="${SCRATCH}/${date_time}"

echo "--> Creating new scratch dir: $SCRATCH_date"
mkdir -p $SCRATCH_date

## Copy bddcml_decomposition project files to temporary (scratch) memory
cd $BDDCML
echo "--> Copying bddcml_decomposition..."
cp -r bddcml_decomposition $SCRATCH_date

## Go to the benchmark directory
cd $SCRATCH_date/bddcml_decomposition/benchmarks/poisson_on_cube/

## Get short commit SHA
commit_short_sha=$(git rev-parse --short HEAD)
echo "--> Current directory: $(/bin/pwd)"
echo "--> Current commit: $commit_short_sha"




# Benchmark script

## Variables
### All Decomposers
declare -a DECOMPOSERS=("PCM_8" "PCM_16" "PCM_32" "ICM_8" "ICM_16" "ICM_32" "CuSolverDnXgetrf" "MAGMAdgetrf_gpu")
### All Solvers
declare -a SOLVERS=("IS_8" "IS_16" "IS_32" "IS_64" "IS_128" "CuBLAStrsm_U" "CuBLAStrsm_L" "CuSolverDnXgetrs" "MAGMAdgetrs_gpu")
### Repository root - assuming that the script was launched from 'benchmarks/poisson_on_cube'
REPO_ROOT=`readlink -f "../../"`
### Benchmarks directory
BENCHMARK_DIR="$REPO_ROOT/benchmarks/poisson_on_cube"
### Executables directory
EXECUTABLES_DIR="$BENCHMARK_DIR/executables"
### Benchmark executable
BENCHMARK_EXECUTABLE="$REPO_ROOT/examples/poisson_on_cube"
### Benchmark name
BENCHMARK_NAME="BDDCML Benchmark: poisson_on_cube"
### Configuration
OUTPUT_DIR="log-files"
LOOPS=10
START_EL_PER_SUB_EDGE=5
END_EL_PER_SUB_EDGE=50
INCREMENT_EL_PER_SUB_EDGE=5
PROCEDURE_TYPE="DECOMPOSERS"

## Functions
function print_time() {
   local secs=$1
   local ELAPSED="Elapsed: $(($secs / 3600))hrs $((($secs / 60) % 60))min $(($secs % 60))sec"
   echo -e "$ELAPSED"
}

function assure_existence_of_output_dir() {
   local output_dir=$1
   local output_dir_abs_path=`readlink -f $output_dir`
   if [ ! -d "$output_dir_abs_path" ]; then
      >&2 echo "-> Warning: Output directory not found: '$output_dir_abs_path'. Creating..."
      mkdir $output_dir_abs_path
   fi
   echo "$output_dir_abs_path"
}

function check_executable_exists() {
   local executable=$1

   if [[ ! -f "$executable" ]]; then
      echo "-!> ERROR: Executable NOT FOUND: $executable"
      echo "Skipping..."
      exit 0
   fi
}

## Finalize variables
### Assure the existence of the output directory
OUTPUT_DIR=$(assure_existence_of_output_dir "$OUTPUT_DIR")

### Check that the executables dir exists and is not empty
if [[ -d "$EXECUTABLES_DIR" ]]; then
   if [ $(ls -A "$EXECUTABLES_DIR" | wc -l) -ne 0 ]; then
      echo "-> Executables directory found: $EXECUTABLES_DIR"
   else
      echo "-!> Executables directory found BUT IT IS EMPTY: $EXECUTABLES_DIR"
      echo "-!> Use the 'prepare_poisson_on_cube_benchmark' script to load it with executables for the benchmark."
      echo "-!> Exiting..."
      exit 1
   fi
else
   echo "-!> Executables directory NOT FOUND: $EXECUTABLES_DIR"
   echo "-!> Create it and use the 'prepare_poisson_on_cube_benchmark' script to load it with executables for the benchmark."
   echo "-!> Exiting..."
   exit 1
fi

## Print Benchmark information
echo -e "======================================== [ START ] $BENCHMARK_NAME\n"
echo "===== Parameters"
echo "LOOPS                         = $LOOPS"
echo "OUTPUT DIRECTORY              = $OUTPUT_DIR"
echo "STARTING EL PER SUB EDGE      = $START_EL_PER_SUB_EDGE"
echo "ENDING EL PER SUB EDGE        = $END_EL_PER_SUB_EDGE"
echo "INCREMENT EL PER SUB EDGE BY  = $INCREMENT_EL_PER_SUB_EDGE"
echo "DECOMPOSERS                   = ${DECOMPOSERS[@]}"
echo "SOLVERS                       = ${SOLVERS[@]}"
echo "BENCHMARKED PROCEDURE TYPE    = $PROCEDURE_TYPE"
echo -e "=====\n"

## Go the the directory containing executables
cd $EXECUTABLES_DIR

## RUN BENCHMARK - DECOMPOSERS
if [[ "$PROCEDURE_TYPE" == "DECOMPOSERS" ]]; then
   echo -e "\n==================== [ START ] $BENCHMARK_NAME - Decomposers\n"

   ### Begin timer
   SECONDS=0

   ### Solvers used for benchmarking decomposers - their performance is not measured
   SOLVER_U=5 # CuBLAStrsm_U
   SOLVER_L=6 # CuBLAStrsm_L
   solver=$SOLVER_U

   num_decomposers=${#DECOMPOSERS[@]}
   for (( decomposer = 0; decomposer < $num_decomposers; decomposer++ ))
   do
      date '+%Y-%m-%d_%H-%M-%S'
      # If the benchmarked decomposer is 6 (CuSolverDnXgetrf) -> change solver to CuBLAStrsm_L
      if [[ $decomposer -eq 6 ]]; then
         solver=$SOLVER_L
      fi

      # If the benchmarked decomposer is 7 (MAGMAdgetrf_gpu) -> set the corresponding solver
      if [[ $decomposer -eq 7 ]]; then
         solver=8 # MAGMAdgetrs_gpu
      fi

      decomposer_name="${DECOMPOSERS[$decomposer]}"
      solver_name="${SOLVERS[$solver]}"
      procedure_combo="${decomposer_name}_${solver_name}"
      executable="$EXECUTABLES_DIR/poisson_on_cube_$procedure_combo"

      # ENABLE - if you want to benchmark only ICM
      # if [[ $decomposer -lt 3 || $decomposer -gt 5 ]]; then
      #    echo -e "\n\n========== [ SKIPPING ] Decomposer ($decomposer: $decomposer_name) - using '$solver: $solver_name' as solver"
      #    continue
      # fi

      echo -e "\n\n========== [ START ] Decomposer ($decomposer: $decomposer_name) - using '$solver: $solver_name' as solver"

      check_executable_exists $executable

      ## TODO: Assert that warmup log file has no errors, if it does -> exit with failure
      # - Maybe enough to check that the results at the end of the log are OK
      # Warmup - redirect output to file
      echo -e "\n===== [ START ] Warmup - Benchmark poisson_on_cube (5 4 2) Decomposer ($decomposer: $decomposer_name) Solver ($solver: $solver_name)"
      log_file="$OUTPUT_DIR/decomposers/raw/$decomposer_name/benchmark_warmup_$decomposer_name.log"
      log_file_dirname=$(dirname "$log_file")
      if [[ ! -d "$log_file_dirname" ]]; then
         mkdir -p "$log_file_dirname"
      fi

      srun --mpi=pmix --cpu-bind=cores $executable 5 4 2 > $log_file 2>&1

      echo -e "===== [ FINISH ] Warmup - Benchmark poisson_on_cube (5 4 2) Decomposer ($decomposer: $decomposer_name) Solver ($solver: $solver_name)"

      for (( elements = $START_EL_PER_SUB_EDGE; elements <= $END_EL_PER_SUB_EDGE; elements += $INCREMENT_EL_PER_SUB_EDGE ))
      do
         config="$elements 4 2"
         configuration_with_underscores="${elements}_4_2"
         echo -e "\n===== [ START ] Configuration ($config)"

         for (( loop = 1; loop <= $LOOPS; loop++ ))
         do
            log_file="$OUTPUT_DIR/decomposers/raw/$decomposer_name/$configuration_with_underscores/poisson_benchmark_loop_${loop}.log"
            echo "Loop: $loop -> $log_file"

            log_file_dirname=$(dirname "$log_file")
            if [[ ! -d "$log_file_dirname" ]]; then
               mkdir -p "$log_file_dirname"
            fi

            if [[ -f "$log_file" ]]; then
               echo "WARNING: deleting an existing log file $log_file"
               rm -f "$log_file"
            fi

            srun --mpi=pmix --cpu-bind=cores $executable $config >> $log_file 2>&1
         done

         echo -e "===== [ FINISH ] Configuration ($config)"
      done

      echo -e "========== [ FINISH ] Decomposer ($decomposer: $decomposer_name) - using '$solver: $solver_name' as solver\n"
   done

   echo -e "==================== [ FINISH ] $BENCHMARK_NAME - Decomposers\n"
   # End timer
   print_time $SECONDS
fi

## RUN BENCHMARK - SOLVERS
if [[ "$PROCEDURE_TYPE" == "SOLVERS" ]]; then
   echo -e "\n==================== [ START ] $BENCHMARK_NAME - Solvers\n"

   ### Begin timer
   SECONDS=0

   ### Decomposers used for benchmarking solvers - their performance is not measured
   DECOMPOSER_U=0 # PCM8
   DECOMPOSER_L=6 # CuSolverDnXgetrf
   decomposer=$DECOMPOSER_U

   num_solvers=${#SOLVERS[@]}
   for (( solver = 0; solver < $num_solvers; solver++ ))
   do
      date '+%Y-%m-%d_%H-%M-%S'
      # If the benchmarked solver is 6, 7 (CuBLAStrsm_L, CuSolverDnXgetrs) -> change decomposer to CuSolverDnXgetrf
      if [[ $solver -eq 6 || $solver -eq 7 ]]; then
         decomposer=$DECOMPOSER_L
      fi

      # If the benchmarked solver is 8 (MAGMAdgetrs_gpu) -> set corresponding decomposer
      if [[ $solver -eq 8 ]]; then
         decomposer=7 # MAGMAdgetrf_gpu
      fi

      solver_name="${SOLVERS[$solver]}"
      decomposer_name="${DECOMPOSERS[$decomposer]}"
      procedure_combo="${decomposer_name}_${solver_name}"
      executable="$EXECUTABLES_DIR/poisson_on_cube_$procedure_combo"

      echo -e "\n========== [ START ] Solver ($solver: $solver_name) - using '$decomposer: $decomposer_name' as decomposer"

      check_executable_exists $executable

      ## TODO: Assert that warmup log file has no errors, if it does -> exit with failure
      # - Maybe enough to check that the results at the end of the log are OK
      # Warmup - redirect output to file
      echo -e "\n===== [ START ] Warmup - Benchmark poisson_on_cube (5 4 2) Solver ($solver: $solver_name) Decomposer ($decomposer: $decomposer_name)"
      log_file="$OUTPUT_DIR/solvers/raw/$solver_name/benchmark_warmup_$solver_name.log"
      log_file_dirname=$(dirname "$log_file")
      if [[ ! -d "$log_file_dirname" ]]; then
         mkdir -p "$log_file_dirname"
      fi

      srun --mpi=pmix --cpu-bind=cores $executable 5 4 2 > $log_file 2>&1
      echo -e "===== [ FINISH ] Warmup - Benchmark poisson_on_cube (5 4 2) Solver ($solver: $solver_name) Decomposer ($decomposer: $decomposer_name)"

      for (( elements = $START_EL_PER_SUB_EDGE; elements <= $END_EL_PER_SUB_EDGE; elements += $INCREMENT_EL_PER_SUB_EDGE ))
      do
         config="$elements 4 2"
         configuration_with_underscores="${elements}_4_2"
         echo -e "\n===== [ START ] Configuration ($config)"

         for (( loop = 1; loop <= $LOOPS; loop++ ))
         do
            log_file="$OUTPUT_DIR/solvers/raw/$solver_name/$configuration_with_underscores/poisson_benchmark_loop_${loop}.log"
            echo "Loop: $loop -> $log_file"

            log_file_dirname=$(dirname "$log_file")
            if [[ ! -d "$log_file_dirname" ]]; then
               mkdir -p "$log_file_dirname"
            fi

            if [[ -f "$log_file" ]]; then
               echo "WARNING: deleting an existing log file $log_file"
               rm -f "$log_file"
            fi

            srun --mpi=pmix --cpu-bind=cores $executable $config > $log_file 2>&1
         done

         echo -e "===== [ FINISH ] Configuration ($config)"
      done

      echo -e "========== [ FINISH ] Solver ($solver: $solver_name) - using '$decomposer: $decomposer_name' as decomposer\n"
   done

   echo -e "==================== [ FINISH ] $BENCHMARK_NAME - Solvers\n"
   # End timer
   print_time $SECONDS
fi




# Post-processing (copy results to permanent storage & clean up)
## Copy logs to permanent storage
benchmark_logs_dir="${PERSONAL}/matrices-logs/logs/bddcml"
benchmark_run_dir="${benchmark_logs_dir}/poc_${START_EL_PER_SUB_EDGE}_to_${END_EL_PER_SUB_EDGE}_${date_time}_${commit_short_sha}_${PROCEDURE_TYPE}"

mkdir $benchmark_run_dir
cp -r $OUTPUT_DIR $benchmark_run_dir

## Clean up after the benchmark - remove the created scratch dir
cd $BDDCML
rm -rf $SCRATCH_date

## Copy the main log file last
cp /home/cejkaluk/benchmarkBDDCML_AMD_ICM_processTol_1_5_to_50.out $benchmark_run_dir
