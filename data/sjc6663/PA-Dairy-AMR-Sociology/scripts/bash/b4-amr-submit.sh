#!/bin/bash
#SBATCH --job-name=Rans4-AMR
#SBATCH -A open
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem-per-cpu=32G
#SBATCH -t 48:00:00
#SBATCH -o Rans4-AMR.out
#SBATCH -e Rans4-AMR.err
#SBATCH --export=ALL
#SBATCH --mail-type=END
#SBATCH --mail-user sjc6663@psu.edu


#write job start time to log file Rans4-AMR.out
echo "Job started at $(date)"

#set variables
READS=/storage/group/evk5387/default/stephanie/ransom-amr++/bovine-host/b4
OUT=/storage/home/sjc6663/scratch/ransom-out/b4

#change to directory where amr++ is installed
cd /storage/home/sjc6663/work/amr++/setup/amrplusplus_v2

#activate bioinfo env for workflow dependencies
module load anaconda3
source activate base

conda activate bioinfo


#specify path for temp files to be written to (in your SCRATCH)
export NXF_WORK=/storage/home/sjc6663/scratch/nf-work-rans/b4


#run amr++
NXF_VER=22.10.5 ../nextflow run main_AmrPlusPlus_v2.nf -profile local --reads "${READS}/R*_merged_R{1,2}.fastq.gz" --output "${OUT}" --threshold 0 --max-threads 20


#write job end time to log file Rans4-AMR.out
echo "Job ended at $(date)"
