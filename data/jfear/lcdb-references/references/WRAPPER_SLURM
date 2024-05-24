#!/usr/bin/bash
#SBATCH --job-name="lcdb-wf"
#SBATCH --partition="norm"
#SBATCH --time=24:00:00

# make logdir
if [[ ! -e slurm_logs ]]; then mkdir -p slurm_logs; fi

# Run snakemake
(conda activate lcdb-references;
    time snakemake \
    -p \
    --directory $PWD \
    -k \
    --rerun-incomplete \
    --jobname "s.{rulename}.{jobid}.sh" \
    -j 999 \
    --cluster-config config/clusterconfig.yaml \
    --verbose \
    --cluster 'sbatch {cluster.prefix} --cpus-per-task={threads}  --output=slurm_logs/{rule}.o.%j --error=slurm_logs/{rule}.e.%j' \
    --use-conda \
    --configfile config/config.yaml \
    --latency-wait=300 \
    ) > "Snakefile.log" 2>&1

SNAKE_PID=$!

finish(){
    echo 'Stopping running snakemake job.'
    kill -SIGINT $SNAKE_PID
    exit 0
}
trap finish SIGTERM

wait $SNAKE_PID
