#!/bin/bash
#SBATCH -J YC-variants
#SBATCH -p intel-e5,amd-ep2
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000M
#SBATCH -o /storage/douyanmeiLab/lujinhong/logs/22_4_16/YC_filter/%x_%J_out.txt

import socket
set +u
module load gcc;
module load jdk/11.0.10;
source /home/douyanmeiLab/lujinhong/miniconda3/etc/profile.d/conda.sh
set -u

##note:
##small amount of jobs: please use debug partition with debug Qos
##large amount of jobs: use intel-e5 and amd-ep2
##amd-ep2-16c: reserved for parallel computing. please make sure the number of CPUs per node could be divided by 16, and use either --exclusive/--ntasks-per-node
##run snakemkake on cluster:
snakemake --unlock --cores 4
#snakemake --rerun-incomplete -j {#jobs} --cluster-config {cluster}.json --cluster "sbatch -p {partition} --account={account} -c {#core} -t #{time} --mem={mem}"
snakemake --rerun-incomplete -j 190 --restart-times 3 --latency-wait 120 --keep-going --cluster "sbatch -p intel-e5,amd-ep2 -q normal -c 1 -o /storage/douyanmeiLab/lujinhong/logs/22_4_16/LZ_filter/%x_%J_out.txt "
