#!/bin/bash
#SBATCH --job-name="SNAKEMAKETEST"
#SBATCH --mem=40g
#SBATCH --partition="norm,ccr"
#SBATCH --time=05-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1

module load python/3.9
module load snakemake/7.32.3
module load singularity/3.10.5

cd $SLURM_SUBMIT_DIR

snakemake -s "/vf/users/CCRCCDI/rawdata/ccrtegs9/snakemake_test/workflow/Snakefile"     --directory "/data/CCRCCDI/rawdata/ccrtegs9/snakemake_test/"     --use-singularity     --use-envmodules     --printshellcmds     --latency-wait 90000     --jobs 1     --configfile "/vf/users/CCRCCDI/rawdata/ccrtegs9/snakemake_test/config/config.yaml"     --cluster-config "/vf/users/CCRCCDI/rawdata/ccrtegs9/snakemake_test/config/cluster.yaml"     --cluster "sbatch --gres {cluster.gres} --cpus-per-task {cluster.threads} -p {cluster.partition} -t {cluster.time} --mem {cluster.mem} --job-name {cluster.name} --output {cluster.output} --error {cluster.error}"     -j 1     --rerun-incomplete     --keep-going     --restart-times 1     --stats "/data/CCRCCDI/rawdata/ccrtegs9/snakemake_test//logs/snakemake.stats"     2>&1 | tee "/data/CCRCCDI/rawdata/ccrtegs9/snakemake_test//logs/snakemake.log"

if [ "$?" -eq "0" ]; then
    snakemake -s "/vf/users/CCRCCDI/rawdata/ccrtegs9/snakemake_test/workflow/Snakefile"     --directory "/data/CCRCCDI/rawdata/ccrtegs9/snakemake_test/"     --report "/data/CCRCCDI/rawdata/ccrtegs9/snakemake_test//logs/runslurm_snakemake_report.html"     --configfile "/vf/users/CCRCCDI/rawdata/ccrtegs9/snakemake_test/config/config.yaml" 
fi

bash <(curl https://raw.githubusercontent.com/CCBR/Tools/master/Biowulf/gather_cluster_stats.sh 2>/dev/null) "/data/CCRCCDI/rawdata/ccrtegs9/snakemake_test//logs/snakemake.log" > "/data/CCRCCDI/rawdata/ccrtegs9/snakemake_test//logs/snakemake.log.HPC_summary.txt"
