#!/bin/bash
#PBS -l mem=160gb,nodes=1:ppn=16,walltime=6:00:00
#PBS -m abe
#PBS -M wyant008@umn.edu
#PBS -q ram256g

module load java

#	Path to the reference
REF=${HOME}/Shared/References/Reference_Sequences/Barley/Morex/barley_RefSeq_v1.0/barley_pseudomolecules_parts.fa

BAM_DIR=/panfs/roc/scratch/wyant008/Barley_NAM_Parents/Realigned_BAMs

#	 Build the sample list
SAMPLE_LIST=($(find ${BAM_DIR} -name '*_realigned.bam'))
echo "$(echo ${SAMPLE_LIST} | wc -l) max task array +1" >&2
CURRENT_SAMPLE=$(basename ${SAMPLE_LIST[${PBS_ARRAYID}]})
SAMPLENAME1=$(echo ${CURRENT_SAMPLE} | cut -f 1 -d '_')
SAMPLENAME2=$(echo ${CURRENT_SAMPLE} | cut -f 2 -d '_')
INFILE=${BAM_DIR}/${CURRENT_SAMPLE}
OUTPUT_DIR=/panfs/roc/groups/9/morrellp/shared/Projects/Barley_NAM_Parents/SNP_calling/GVCFs

mkdir -p ${OUTPUT_DIR}

#	HaplotypeCaller options:
#		-R Reference
#		-L regions.bed: operate only over supplied regions
#		-I Input file
#		-o Output file
#		--genotyping_mode DISCOVERY: call new variants
#		--emitRefConfidence GVCF: output GLs instead of SNPs, useful for
#		batch calling many samples later
#		--heterozygosity 0.008: Use a prior on nucleotide diversity of 0.008/bp
#		-nct 4: use 4 CPU threads

GATK=/panfs/roc/groups/9/morrellp/shared/Software/GATK-3.6/GenomeAnalysisTK.jar

export _JAVA_OPTIONS="-Xmx63g"
java -jar ${GATK}\
	-T HaplotypeCaller\
	-R ${REF}\
	-I ${INFILE}\
	-o ${OUTPUT_DIR}/${SAMPLENAME1}_${SAMPLENAME2}_RawGLs.g.vcf\
	-nct 16\
	--genotyping_mode DISCOVERY\
	--heterozygosity 0.008\
	--emitRefConfidence GVCF\
	-variant_index_type LINEAR\
	-variant_index_parameter 128000
