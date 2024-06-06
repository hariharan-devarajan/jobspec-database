#! /usr/bin/env bash

#BSUB -J cellranger
#BSUB -o logs/cellranger_%J.out
#BSUB -e logs/cellranger_%J.err
#BSUB -R "select[mem>4] rusage[mem=4]"
#BSUB -q rna

set -o nounset -o pipefail -o errexit -x

module load cellranger/5.0.1

mkdir -p logs

run_snakemake() {
    local config_file=$1
    
    drmaa_args='
        -o {log.out}
        -e {log.err}
        -J {params.job_name} 
        -R "{params.memory} span[hosts=1]"
        -R "select[hname!=compute16]"
        -R "select[hname!=compute19]"
        -n {threads} '

    snakemake \
        --snakefile Snakefile \
        --drmaa "$drmaa_args" \
        --jobs 300 \
        --latency-wait 60 \
        --configfile $config_file
}

# 24 hpi run 2021-01-08
run_snakemake src/configs/2021-01-08.yaml

# 8 hpi run 2021-04-16
run_snakemake src/configs/2021-04-16.yaml

# 8 hpi CHIKV enrichment 2021-07-16
run_snakemake src/configs/2021-07-16.yaml

