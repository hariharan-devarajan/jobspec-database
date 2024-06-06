#!/bin/bash
#BSUB -J "freebayes[1-97]%20"
#BSUB -W 04:00
#BSUB -n 1
#BSUB -R "rusage[mem=500]"

##Variant calling producing for each BAM file a VCF file including variant and invariant sites

Ref=/cluster/scratch/mleeman/freebayes/whitefish_reordered.fasta
out=/cluster/scratch/mleeman/freebayes/allLakesSingle
in=/cluster/scratch/mleeman/bwa/allLakesSingle

module load gcc/4.8.2 gdc perl/5.18.4 samtools/1.10

samtools faidx ${Ref}

module load gcc/4.8.2 gdc python/2.7.11 freebayes/1.3.4

IDX=$LSB_JOBINDEX
name=`sed -n ${IDX}p < ${out}/sample.list`

freebayes -f ${Ref} ${in}/${name}_sort_20_nodup.bam -p 1 \
          --limit-coverage 25 --min-mapping-quality 30 --min-base-quality 20 \
          --hwe-priors-off --report-monomorphic --vcf ${out}/${name}.vcf
