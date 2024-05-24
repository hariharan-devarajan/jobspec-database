#!/bin/bash

### Optional. Set the job name
#SBATCH --job-name=SLURM_MetaP
### REQUIRED. Specify the PI group for this job
#SBATCH --account=
### Optional. Request email when job begins and ends
### SBATCH --mail-type=ALL
### Optional. Specify email address to use for notification
### SBATCH --mail-user=
### REQUIRED. Set the partition for your job.
#SBATCH --partition=
### REQUIRED. Set the number of cores that will be used for this job.
#SBATCH --ntasks=5
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=60gb
### REQUIRED. Specify the time required for this job, hhh:mm:ss
#SBATCH --time=72:00:00

source ~/.bashrc
source activate metaphlan3

cd Metaphlan3_snakemake  

#dry run
#snakemake -n

#run in cluster
snakemake --cluster "sbatch -A {cluster.group} -p {cluster.partition} -n {cluster.n} -t {cluster.time} --mem={cluster.m}"  --cluster-config config/cluster.yaml -j 60 --latency-wait 15


