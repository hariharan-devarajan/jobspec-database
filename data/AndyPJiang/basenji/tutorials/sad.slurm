#!/bin/bash
# 24734562
#SBATCH --nodes 16
#SBATCH --account=punim0614
#SBATCH --partition physical
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --job-name=sad
#SBATCH --mem=100000

#load modules
module load Python/3.7.1-GCC-6.2.0
module load Tensorflow/1.15.0-GCC-6.2.0-Python-3.7.1-GPU
export BASENJIDIR=/data/gpfs/projects/punim0614/andy/basenji21/basenji
export PATH=$BASENJIDIR/bin:$PATH
export PYTHONPATH=$BASENJIDIR/bin:$PYTHONPATH
source ${HOME}/basenji/bin/activate

#execute relevant python script
python ../bin/basenji_sad.py --cpu -f data/hg19.ml.fa -o output/rfx6_sad_all --rc --shift "1,0,-1" -t data/lcl_wigs.txt models/params.txt models/lcl/model_human.tf data/dsQTL.eval.flipped.vcf

