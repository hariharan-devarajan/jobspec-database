#!/bin/bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=00:30:00
#SBATCH --mem=16GB
#SBATCH -o /home/a1671704/fastdir/Data_TigerSnake/3_transcriptomeAssembly/cdHITandDeNovo/qualityAssessment/slurm/ExN50_%j.out
#SBATCH -e /home/a1671704/fastdir/Data_TigerSnake/3_transcriptomeAssembly/cdHITandDeNovo/qualityAssessment/slurm/ExN50_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=a1671704@student.adelaide.edu.au

TRINITYDIR=/apps/software/Trinity/2.5.1-foss-2016b/trinityrnaseq-Trinity-v2.5.1/util/misc
MATRIXDIR=/home/a1671704/fastdir/Data_TigerSnake/3_transcriptomeAssembly/cdHITandDeNovo/downstreamAnalysis
ASSEMDIR=/home/a1671704/fastdir/Data_TigerSnake/3_transcriptomeAssembly/cdHITandDeNovo
OUTDIR=/home/a1671704/fastdir/Data_TigerSnake/3_transcriptomeAssembly/cdHITandDeNovo/qualityAssessment

${TRINITYDIR}/contig_ExN50_statistic.pl \
  ${MATRIXDIR}/CDHITKallistoAbundance.isoform.TMM.EXPR.matrix \
  ${ASSEMDIR}/collaspedTrinityDeNovo90 | \
  tee ${OUTDIR}/ExN90stats.txt
