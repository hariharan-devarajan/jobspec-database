#!/bin/sh
#SBATCH --account=gdkendalllab
#SBATCH --error=slurmOut/count-%j.txt
#SBATCH --output=slurmOut/count-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name count
#SBATCH --wait
#SBATCH --mail-user=matthew.cannon@nationwidechildrens.org
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80

set -e ### stops bash script if line ends with error

echo ${HOSTNAME} ${SLURM_ARRAY_TASK_ID}

module purge

module load GCC/9.3.0 \
            SAMtools/1.10

featureCounts \
  -T 10 \
  -a ~/analyses/kendall/refs/annotations/danRer11.ensGene_WithERCC.gtf.gz \
  -o output/geneCounts/combined.txt \
  -p \
  --countReadPairs \
  -s 0 \
  output/aligned/*.bam

