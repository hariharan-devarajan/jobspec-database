#!/bin/bash

set -eo pipefail
module purge

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <RUNMODE> <WORKDIR>"
    exit 1
fi

# Set input arguments
RUNMODE=$1
WORKDIR=$2
ENTRY=$3

# Ensure RUNMODE is provided
if [ -z "$RUNMODE" ]; then
    echo "RUNMODE argument is missing"
    exit 1
fi

# Ensure WORKDIR is provided and exists
if [ -z "$WORKDIR" ]; then
    echo "WORKDIR argument is missing"
    exit 1
elif [ ! -d "$WORKDIR" ]; then
    echo "WORKDIR does not exist: $WORKDIR"
    exit 1
fi

echo "RUNMODE: $RUNMODE"
echo "WORKDIR: $WORKDIR"

# Set variables
PIPELINE_HOME=$(readlink -f "$(dirname "$0")")
SNAKEFILE="${PIPELINE_HOME}/workflow/Snakefile"
PARTITIONS="norm,ccr"
PYTHON_VERSION="python/3.9"
SNAKEMAKE_VERSION="snakemake/7.32.3"
SINGULARITY_VERSION="singularity/3.10.5"

# Load necessary modules
module load "$SNAKEMAKE_VERSION"

# Define essential files and folders
ESSENTIAL_CONFIGS="config/config.yaml config/fqscreen_config.conf config/multiqc_config.yaml config/cluster.yaml config/tools.yaml"
ESSENTIAL_MANIFESTS="manifest/sample_manifest.csv"
ESSENTIAL_FOLDERS="workflow/scripts"

# Initialize workdir if RUNMODE is "init"
##if [ "$RUNMODE" == "init" ]; then
##    init
## elif 
if [ "$RUNMODE" == "local" ]; then
    # Run Snakemake locally
    echo "local mode"
    echo "test mode"

    snakemake -s "$SNAKEFILE" \
    --directory "$WORKDIR" \
    --printshellcmds \
    --use-singularity \
    --use-envmodules \
    --jobs 1 \
    --latency-wait 90000 -j 1 \
    --configfile "${WORKDIR}/config/config.yaml" \
    --cores all \
    --stats "${WORKDIR}/logs/snakemake.stats" \
    2>&1 | tee "${WORKDIR}/logs/snakemake.log"


    echo "call mode"
    # Generate report if successful
    if [ "$?" -eq "0" ]; then
        snakemake -s "$SNAKEFILE" \
            --report "${WORKDIR}/logs/runlocal_snakemake_report.html" \
            --directory "$WORKDIR" \
            --configfile "${WORKDIR}/config/config.yaml" 
    fi

elif [ "$RUNMODE" == "slurm" ]; then
  
    cat > "${WORKDIR}/submit_script.sbatch" << EOF
#!/bin/bash
#SBATCH --job-name="SNAKEMAKETEST"
#SBATCH --mem=40g
#SBATCH --partition="$PARTITIONS"
#SBATCH --time=05-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1

module load $PYTHON_VERSION
module load $SNAKEMAKE_VERSION
module load $SINGULARITY_VERSION

cd \$SLURM_SUBMIT_DIR

snakemake -s "$SNAKEFILE" \
    --directory "$WORKDIR" \
    --use-singularity \
    --use-envmodules \
    --printshellcmds \
    --latency-wait 90000 \
    --jobs 1 \
    --configfile "${PIPELINE_HOME}/config/config.yaml" \
    --cluster-config "${PIPELINE_HOME}/config/cluster.yaml" \
    --cluster "sbatch --gres {cluster.gres} --cpus-per-task {cluster.threads} -p {cluster.partition} -t {cluster.time} --mem {cluster.mem} --job-name {cluster.name} --output {cluster.output} --error {cluster.error}" \
    -j 1 \
    --rerun-incomplete \
    --keep-going \
    --restart-times 1 \
    --stats "${WORKDIR}/logs/snakemake.stats" \
    2>&1 | tee "${WORKDIR}/logs/snakemake.log"

if [ "\$?" -eq "0" ]; then
    snakemake -s "$SNAKEFILE" \
    --directory "$WORKDIR" \
    --report "${WORKDIR}/logs/runslurm_snakemake_report.html" \
    --configfile "${PIPELINE_HOME}/config/config.yaml" 
fi

bash <(curl https://raw.githubusercontent.com/CCBR/Tools/master/Biowulf/gather_cluster_stats.sh 2>/dev/null) "${WORKDIR}/logs/snakemake.log" > "${WORKDIR}/logs/snakemake.log.HPC_summary.txt"
EOF

    sbatch "${WORKDIR}/submit_script.sbatch"

else
    # Run Snakemake for unlock and dryrun modes
    echo "--$RUNMODE"
    snakemake -s "$SNAKEFILE" \
        "--$RUNMODE" \
        --directory "$WORKDIR" \
        --use-envmodules \
        --printshellcmds \
        --latency-wait 120 \
        --jobs 1 \
        --configfile "${PIPELINE_HOME}/config/config.yaml" \
        --cluster-config "${PIPELINE_HOME}/config/cluster.yaml" \
        --cluster "sbatch --gres {cluster.gres} --cpus-per-task {cluster.threads} -p {cluster.partition} -t {cluster.time} --mem {cluster.mem} --job-name {cluster.name} --output {cluster.output} --error {cluster.error}" \
        -j 1 \
        --rerun-incomplete \
        --keep-going \
        --touch \
        --stats "${WORKDIR}/logs/snakemake.stats"
fi
