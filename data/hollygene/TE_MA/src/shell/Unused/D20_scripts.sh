#PBS -S /bin/bash
#PBS -q highmem_q
#PBS -N piped_command
#PBS -l nodes=1:ppn=1:HIGHMEM
#PBS -l walltime=96:00:00
#PBS -l mem=250gb
#PBS -M hcm14449@uga.edu
#PBS -m abe

#S paradoxus TE MA Quality Control and Mutation Calling pipeline

#location of current update of muver
# muver_module="muver/0.1.0-foss-2016b-Python-2.7.14-20190318"
#location of trimgalore moedule
# trimgalore_module="Trim_Galore/0.4.5-foss-2016b"
#location of fastqc module
# fastqc_module="FastQC/0.11.8-Java-1.8.0_144"
#location of BWA module
bwa_module="BWA/0.7.15-foss-2016b"
#location of samtools module
samtools_module="SAMtools/1.6-foss-2016b"
#location of bedtools module
bedtools_module="BEDTools/2.28.0-foss-2018a"
#location of python module
python_module="Python/3.5.2-foss-2016b"
#location of picard module
picard_module="picard/2.21.6-Java-11"
#location of GATK module
GATK_module="GATK/4.0.3.0-Java-1.8.0_144"
#deeptools location
deeptools_module="deepTools/3.2.1-foss-2018a-Python-3.6.4"
#location of bam to bigwig script
bamToBigWig="/scratch/hcm14449/TE_MA_Paradoxus/bedGraphToBigWigScript/file_to_bigwig_pe.py"
#location of reference genome to be used
ref_genome="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/genome.337.fasta"
#directory reference genome is located in
ref_genome_dir="/scratch/hcm14449/TE_MA_Paradoxus/ref_genome/paradoxus/337Ref/"
#where should the output be sent
output_directory="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/Out/D20"
# mkdir $output_directory
#location of data to be used in the analysis
raw_data="/scratch/hcm14449/TE_MA_Paradoxus/Illumina_Data/AllFastas"
mcc_bams="/scratch/jc33471/paradoxusHolly/run0217all"
mcc_bam_indiv="/scratch/jc33471/paradoxusHolly/run0217all/out/Spar"

cd ${output_directory}
# rm *

module load ${picard_module}
module load ${bwa_module}
module load ${samtools_module}
module load ${GATK_module}

# #######################################################################################
# # create a uBAM file
# #######################################################################################
#
# for file in ${output_directory}/D20/*_R1_001.fastq.gz
#
# do
#   FBASE=$(basename $file _R1_001.fastq.gz)
#   BASE=${FBASE%_R1_001.fastq.gz}
# java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.21.6-Java-11" -jar  \
#   /usr/local/apps/eb/picard/2.21.6-Java-11/picard.jar FastqToSam \
#       FASTQ=${output_directory}/D20/${BASE}_R1_001.fastq.gz \
#       FASTQ2=${output_directory}/D20/${BASE}_R2_001.fastq.gz  \
#       OUTPUT=${output_directory}/D20/${BASE}_fastqtosam.bam \
#       READ_GROUP_NAME=${BASE} \
#       SAMPLE_NAME=${BASE} \
#       PLATFORM=illumina \
#       SEQUENCING_CENTER=GGBC
# done
#
#
#
# for file in ${output_directory}/D20/*R1.fq.gz
#
# do
#   FBASE=$(basename $file R1.fq.gz)
#   BASE=${FBASE%R1.fq.gz}
#   java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.21.6-Java-11" -jar  \
#     /usr/local/apps/eb/picard/2.21.6-Java-11/picard.jar FastqToSam \
#         FASTQ=${output_directory}/D20/${BASE}R1.fq.gz \
#         FASTQ2=${output_directory}/D20/${BASE}R2.fq.gz  \
#         OUTPUT=${output_directory}/D20/${BASE}_fastqtosam.bam \
#         READ_GROUP_NAME=${BASE} \
#         SAMPLE_NAME=${BASE} \
#         PLATFORM=illumina \
#         SEQUENCING_CENTER=GGBC
#   done
#
# for file in ${raw_data}/*_R1_001.fastq
#
# do
#   FBASE=$(basename $file _R1_001.fastq)
#   BASE=${FBASE%_R1_001.fastq}
# 	OUT="${BASE}_FastqToSam.sh"
# 	echo "#!/bin/bash" > ${OUT}
# 	echo "#PBS -N ${BASE}_FastqToSam" >> ${OUT}
# 	echo "#PBS -l walltime=12:00:00" >> ${OUT}
# 	echo "#PBS -l nodes=1:ppn=1:AMD" >> ${OUT}
# 	echo "#PBS -q batch" >> ${OUT}
# 	echo "#PBS -l mem=40gb" >> ${OUT}
# 	echo "" >> ${OUT}
# 	echo "cd ${raw_data}" >> ${OUT}
# 	echo "module load ${picard_module}" >> ${OUT}
#   echo "module load ${bwa_module}" >> ${OUT}
#   echo "module load ${samtools_module}" >> ${OUT}
#   echo "module load ${GATK_module}" >> ${OUT}
# 	echo "" >> ${OUT}
#   echo "java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
#   /usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar FastqToSam \
#       FASTQ=${raw_data}/${BASE}_R1_001.fastq \
#       FASTQ2=${raw_data}/${BASE}_R2_001.fastq  \
#       OUTPUT=${raw_data}/${BASE}_fastqtosam.bam \
#       READ_GROUP_NAME=${BASE} \
#       SAMPLE_NAME=${BASE} \
#       LIBRARY_NAME=D0 \
#       PLATFORM=illumina \
#       SEQUENCING_CENTER=GGBC" >> ${OUT}
# 	qsub ${OUT}
# done
#######################################################################################
# mark Illumina adapters
#######################################################################################

# mkdir ${raw_data}/TMP
#
# for file in ${output_directory}/D20/${BASE}_fastqtosam.bam
#
# do
#
# FBASE=$(basename $file _fastqtosam.bam)
# BASE=${FBASE%_fastqtosam.bam}
#
# java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
# /usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar MarkIlluminaAdapters \
# I=${output_directory}/D20/${BASE}_fastqtosam.bam \
# O=${output_directory}/D20/${BASE}_markilluminaadapters.bam \
# M=${output_directory}/D20/${BASE}_markilluminaadapters_metrics.txt \
# TMP_DIR=${output_directory}/D20/TMP \
# USE_JDK_DEFLATER=true \
# USE_JDK_INFLATER=true
#
# done


# for file in ${raw_data}/*_fastqtosam.bam
#
# do
#   FBASE=$(basename $file _fastqtosam.bam)
#   BASE=${FBASE%_fastqtosam.bam}
# 	OUT="${BASE}_MarkIlluminaAdapters.sh"
# 	echo "#!/bin/bash" > ${OUT}
# 	echo "#PBS -N ${BASE}_MarkIlluminaAdapters" >> ${OUT}
# 	echo "#PBS -l walltime=12:00:00" >> ${OUT}
# 	echo "#PBS -l nodes=1:ppn=1:AMD" >> ${OUT}
# 	echo "#PBS -q batch" >> ${OUT}
# 	echo "#PBS -l mem=20gb" >> ${OUT}
# 	echo "" >> ${OUT}
# 	echo "cd ${raw_data}" >> ${OUT}
# 	echo "module load ${picard_module}" >> ${OUT}
#   echo "module load ${bwa_module}" >> ${OUT}
#   echo "module load ${samtools_module}" >> ${OUT}
#   echo "module load ${GATK_module}" >> ${OUT}
# 	echo "" >> ${OUT}
#   echo "mkdir ${raw_data}/TMP" >> ${OUT}
#   echo "java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
#   /usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar MarkIlluminaAdapters \
#   I=${raw_data}/${BASE}_fastqtosam.bam \
#   O=${raw_data}/${BASE}_markilluminaadapters.bam \
#   M=${raw_data}/${BASE}_markilluminaadapters_metrics.txt \
#   TMP_DIR=${raw_data}/TMP" >> ${OUT}
# 	qsub ${OUT}
# done

# for file in ${raw_data}/*_markilluminaadapters.bam
#
# do
#   FBASE=$(basename $file _markilluminaadapters.bam)
#   BASE=${FBASE%_markilluminaadapters.bam}
# 	OUT="${BASE}_SamToFastq.sh"
# 	echo "#!/bin/bash" > ${OUT}
# 	echo "#PBS -N ${BASE}_SamToFastq" >> ${OUT}
# 	echo "#PBS -l walltime=12:00:00" >> ${OUT}
# 	echo "#PBS -l nodes=1:ppn=1:AMD" >> ${OUT}
# 	echo "#PBS -q batch" >> ${OUT}
# 	echo "#PBS -l mem=10gb" >> ${OUT}
# 	echo "" >> ${OUT}
# 	echo "cd ${raw_data}" >> ${OUT}
# 	echo "module load ${picard_module}" >> ${OUT}
#   echo "module load ${bwa_module}" >> ${OUT}
#   echo "module load ${samtools_module}" >> ${OUT}
#   echo "module load ${GATK_module}" >> ${OUT}
# 	echo "" >> ${OUT}
#   echo "java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
#   /usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar SamToFastq \
#   I=${raw_data}/${BASE}_markilluminaadapters.bam \
#   FASTQ=${raw_data}/${BASE}_samtofastq_interleaved.fq \
#   CLIPPING_ATTRIBUTE=XT \
#   CLIPPING_ACTION=2 \
#   INTERLEAVE=true \
#   NON_PF=true \
#   TMP_DIR=${raw_data}/TMP" >> ${OUT}
# 	qsub ${OUT}
# done


## Piped command: SamToFastq, then bwa mem, then MergeBamAlignment
# for file in ${raw_data}/*_markilluminaadapters.bam
#
# do
#
# FBASE=$(basename $file _markilluminaadapters.bam)
# BASE=${FBASE%_markilluminaadapters.bam}
# OUT="${BASE}_piped.sh"
# echo "#!/bin/bash" > ${OUT}
# echo "#PBS -N ${BASE}_piped" >> ${OUT}
# echo "#PBS -l walltime=12:00:00" >> ${OUT}
# echo "#PBS -l nodes=1:ppn=1:AMD" >> ${OUT}
# echo "#PBS -q batch" >> ${OUT}
# echo "#PBS -l mem=30gb" >> ${OUT}
# echo "" >> ${OUT}
# echo "cd ${raw_data}" >> ${OUT}
# echo "module load ${picard_module}" >> ${OUT}
# echo "module load ${bwa_module}" >> ${OUT}
# echo "module load ${samtools_module}" >> ${OUT}
# echo "module load ${GATK_module}" >> ${OUT}
# echo "" >> ${OUT}
# echo "java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
# /usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar SamToFastq \
# I=${raw_data}/${BASE}_markilluminaadapters.bam \
# FASTQ=/dev/stdout \
# CLIPPING_ATTRIBUTE=XT CLIPPING_ACTION=2 INTERLEAVE=true NON_PF=true \
# TMP_DIR=${raw_data}/TMP | \
# bwa mem -M -t 7 -p ${ref_genome} /dev/stdin| \
# java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
# /usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar MergeBamAlignment \
# ALIGNED_BAM=/dev/stdin \
# UNMAPPED_BAM=${raw_data}/${BASE}_fastqtosam.bam \
# OUTPUT=${raw_data}/${BASE}_pipedNewRef.bam \
# R=${ref_genome} CREATE_INDEX=true ADD_MATE_CIGAR=true \
# CLIP_ADAPTERS=false CLIP_OVERLAPPING_READS=true \
# INCLUDE_SECONDARY_ALIGNMENTS=true MAX_INSERTIONS_OR_DELETIONS=-1 \
# PRIMARY_ALIGNMENT_STRATEGY=MostDistant ATTRIBUTES_TO_RETAIN=XS \
# TMP_DIR=${raw_data}/TMP" >> ${OUT}
# qsub ${OUT}
#
# done

# for file in ${output_directory}/D20/*_markilluminaadapters.bam
#
# do
#
# FBASE=$(basename $file _markilluminaadapters.bam)
# BASE=${FBASE%_markilluminaadapters.bam}
# OUT="${BASE}_pipedNewRef.sh"
# echo "#!/bin/bash" >> ${OUT}
# echo "#PBS -N ${BASE}_pipedNewRef" >> ${OUT}
# echo "#PBS -l walltime=12:00:00" >> ${OUT}
# echo "#PBS -l nodes=1:ppn=1:HIGHMEM" >> ${OUT}
# echo "#PBS -q highmem_q" >> ${OUT}
# echo "#PBS -l mem=150gb" >> ${OUT}
# echo "" >> ${OUT}
# echo "cd ${output_directory}/D20" >> ${OUT}
# echo "module load ${picard_module}" >> ${OUT}
# echo "module load ${bwa_module}" >> ${OUT}
# echo "module load ${samtools_module}" >> ${OUT}
# echo "module load ${GATK_module}" >> ${OUT}
# echo "" >> ${OUT}
# echo "java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
#   /usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar SamToFastq \
#   I=${output_directory}/D20/${BASE}_markilluminaadapters.bam \
#   FASTQ=/dev/stdout \
#   CLIPPING_ATTRIBUTE=XT CLIPPING_ACTION=2 INTERLEAVE=true NON_PF=true \
#   TMP_DIR=${output_directory}/D20/TMP | \
#   bwa mem -M -t 7 -p ${ref_genome} /dev/stdin| \
#   java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144" -jar  \
#   /usr/local/apps/eb/picard/2.4.1-Java-1.8.0_144/picard.jar MergeBamAlignment \
#   ALIGNED_BAM=/dev/stdin \
#   UNMAPPED_BAM=${output_directory}/D20/${BASE}_fastqtosam.bam \
#   OUTPUT=${output_directory}/D20/${BASE}_pipedNewRef.bam \
#   R=${ref_genome} CREATE_INDEX=true ADD_MATE_CIGAR=true \
#   CLIP_ADAPTERS=false CLIP_OVERLAPPING_READS=true \
#   INCLUDE_SECONDARY_ALIGNMENTS=true MAX_INSERTIONS_OR_DELETIONS=-1 \
#   PRIMARY_ALIGNMENT_STRATEGY=MostDistant ATTRIBUTES_TO_RETAIN=XS \
#   TMP_DIR=${output_directory}/D20/TMP" >> ${OUT}
# qsub ${OUT}
#
# done
#######################################################################################
# # works: aligns samples to reference genome. Output is a .sam file
# #######################################################################################
#
# module load ${bwa_module}
# #
# #  #index the ref genome
# bwa index ${ref_genome}
# #
# for file in ${trimmed_data}/*_R1_001_trimmed.fq
#
# do
#
# FBASE=$(basename $file _R1_001_trimmed.fq)
# BASE=${FBASE%_R1_001_trimmed.fq}
#
# bwa mem -p -M -t 12 ${ref_genome} ${trimmed_data}/${BASE}_R1_001_trimmed.fq ${trimmed_data}/${BASE}_R2_001_trimmed.fq > ${output_directory}/${BASE}_aln.sam
#
# done
# # #########################################################################################
# # #samtools: converts sam files to bam files and sorts them
# # #########################################################################################
# #
# module load ${samtools_module}
#
#
# #convert sam files to bam files
# for file in ${output_directory}/*_aln.sam
#
# do
#
# FBASE=$(basename $file _aln.sam)
# BASE=${FBASE%_aln.sam}
#
# samtools view -bt ${ref_genome_dir}/*.fai \
# ${output_directory}/${BASE}_aln.sam \
#   > ${output_directory}/${BASE}.bam
#
# done
#
# for file in ${output_directory}/*.bam
#
# do
#
# FBASE=$(basename $file .bam)
# BASE=${FBASE%.bam}
#
# samtools sort -@ 12 -o ${output_directory}/${BASE}.sorted.bam \
#    ${output_directory}/${BASE}.bam
#
# done
#
# # ############################
# # ### index the bam files
# # ############################
#
# for file in ${output_directory}/*.sorted.bam
#
# do
#
# FBASE=$(basename $file .sorted.bam)
# BASE=${FBASE%.sorted.bam}
#
# samtools index -@ 12 ${output_directory}/${BASE}.sorted.bam
#
# done
#
# # ###################################################################################################
# # ## Picard to mark duplicates
# # ###################################################################################################
# #
# module load ${picard_module}
#
#
# for file in ${output_directory}/*.sorted.bam
#
# do
#
# FBASE=$(basename $file .sorted.bam)
# BASE=${FBASE%.sorted.bam}
#
# time java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
# /usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar ValidateSamFile \
#       I=${output_directory}/${BASE}.sorted.bam \
#       MODE=VERBOSE
#
# done
# ###################################################################################################
# for file in ${output_directory}/*.sorted.bam
#
# do
#
# FBASE=$(basename $file .sorted.bam)
# BASE=${FBASE%.sorted.bam}
#
# time java -Xmx20g -classpath "/usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144" -jar  \
# /usr/local/apps/eb/picard/2.16.0-Java-1.8.0_144/picard.jar MarkDuplicates \
# REMOVE_DUPLICATES=TRUE \
# I=${output_directory}/${BASE}.sorted.bam \
# O=${output_directory}/${BASE}_removedDuplicates.bam \
# M=${output_directory}/${BASE}_removedDupsMetrics.txt
#
# done
#
# ###################################################################################################
# # Using GATK HaplotypeCaller in GVCF mode
# # apply appropriate ploidy for each sample
# # will need to do this separtely for haploid and diploid samples
# ###################################################################################################
# #
# ###################################################################################################
# #
# module load ${GATK_module}

# D20 samples
# for file in ${raw_data}/${BASE}*_piped.bam
#
# do
#
# FBASE=$(basename $file _piped.bam)
# BASE=${FBASE%_piped.bam}
#
# time gatk HaplotypeCaller \
#      -R ${ref_genome} \
#      -ERC GVCF \
#      -I ${raw_data}/${BASE}_piped.bam \
#      -ploidy 2 \
#      -O ${output_directory}/${BASE}_variants.g.vcf
#
# done

# for file in ${raw_data}/${BASE}*_piped.bam
#
# do
#
# FBASE=$(basename $file _piped.bam)
# BASE=${FBASE%_piped.bam}
# OUT="${BASE}_HC.sh"
# echo "#!/bin/bash" >> ${OUT}
# echo "#PBS -N ${BASE}_HC" >> ${OUT}
# echo "#PBS -l walltime=72:00:00" >> ${OUT}
# echo "#PBS -l nodes=1:ppn=1:HIGHMEM" >> ${OUT}
# echo "#PBS -q highmem_q" >> ${OUT}
# echo "#PBS -l mem=200gb" >> ${OUT}
# echo "" >> ${OUT}
# echo "module load ${GATK_module}" >> ${OUT}
# echo "" >> ${OUT}
# echo "time gatk HaplotypeCaller \
# -R ${ref_genome} \
# -ERC GVCF \
# -I ${raw_data}/${BASE}_piped.bam \
# -ploidy 2 \
# -O ${output_directory}/${BASE}_variants.g.vcf" >> ${OUT}
# qsub ${OUT}
#
# done

# module load GATK/4.0.3.0-Java-1.8.0_144
#
time gatk HaplotypeCaller \
     -R ${ref_genome} \
     -ERC GVCF \
     -I ${output_directory}/bams/pipedNewRef/D20-A_pipedNewRef.bam \
     -ploidy 2 \
     -O ${output_directory}/gVCFs/D20-A_R_variants.g.vcf

#
# ###################################################################################################
### Combine gVCFs before joint genotyping
# ###################################################################################################


time gatk CombineGVCFs \
 -O ${output_directory}/gVCFs/D20_cohortNewRef.g.vcf \
 -R ${ref_genome} \
 --variant ${output_directory}/gVCFs/D20-A_R_variants.g.vcf \
 --variant ${output_directory}/gVCFs/D20-10_variants.g.vcf \
 --variant ${output_directory}/gVCFs/D20-11_variants.g.vcf \
 --variant ${output_directory}/gVCFs/D20-12_variants.g.vcf \
 --variant ${output_directory}/gVCFs/D20-13_variants.g.vcf \
 --variant ${output_directory}/gVCFs/D20-14_variants.g.vcf \
 --variant ${output_directory}/gVCFs/D20-15_variants.g.vcf \
 --variant ${output_directory}/gVCFs/D20-16_variants.g.vcf




###################################################################################################
### Jointly genotype 8 random samples to identify consensus sequences
###################################################################################################

time gatk GenotypeGVCFs \
        -R ${ref_genome} \
        --variant ${output_directory}/gVCFs/D20_cohortNewRef.g.vcf \
        -O ${output_directory}/gVCFs/D20_variants_8SamplesNewRef.vcf


# ###################################################################################################
# ## Recalibrate base quality scores in all samples to mask any likely consensus variants
# ###################################################################################################
#
for file in ${output_directory}/bams/_removedDuplicates/${BASE}*_removedDuplicates.bam

do

FBASE=$(basename $file _removedDuplicates.bam)
BASE=${FBASE%_removedDuplicates.bam}

time gatk BaseRecalibrator \
-I ${output_directory}/bams/_removedDuplicates/${BASE}_removedDuplicates.bam \
--known-sites ${output_directory}/gvcfs/D20_variants_8SamplesNewRef.vcf \
-O ${output_directory}/recalibrated/${BASE}_recal_data.table \
-R ${ref_genome}

done


# ###################################################################################################
# ## Apply BQSR to bam files
# ###################################################################################################
#



for file in ${output_directory}/bams/_removedDuplicates/${BASE}*_removedDuplicates.bam

do
  FBASE=$(basename $file _removedDuplicates.bam)
  BASE=${FBASE%_removedDuplicates.bam}
  OUT="${BASE}_BQSR.sh"
  echo "#!/bin/bash" >> ${OUT}
  echo "#PBS -N ${BASE}_HC" >> ${OUT}
  echo "#PBS -l walltime=72:00:00" >> ${OUT}
  echo "#PBS -l nodes=1:ppn=1:HIGHMEM" >> ${OUT}
  echo "#PBS -q highmem_q" >> ${OUT}
  echo "#PBS -l mem=200gb" >> ${OUT}
  echo "" >> ${OUT}
  echo "cd ${output_directory}" >> ${OUT}
  echo "module load ${GATK_module}" >> ${OUT}
  echo "" >> ${OUT}
  echo "gatk ApplyBQSR \
  -R ${ref_genome} \
  -I ${output_directory}/bams/_removedDuplicates/${BASE}_removedDuplicates.bam \
  -bqsr ${output_directory}/recalibrated/${BASE}_recal_data.table \
  -O ${output_directory}/bams/recalibrated/${BASE}_recalibratedNewRef.bam" >> ${OUT}
  qsub ${OUT}

done


  # ###################################################################################################
  ### Run HaplotypeCaller again on recalibrated samples
  # ###################################################################################################
  # ###################################################################################################
  # #
        # module load ${GATK_module}

### N E X T ####

#
# for file in ${output_directory}/bams/recalibrated/${BASE}*_recalibratedNewRef.bam
#
# do
#
# FBASE=$(basename $file _recalibratedNewRef.bam)
# BASE=${FBASE%_recalibratedNewRef.bam}
# OUT="${BASE}_HapCaller.sh"
#
# echo "#!/bin/bash" >> ${OUT}
# echo "#PBS -N ${BASE}_HapCaller" >> ${OUT}
# echo "#PBS -l walltime=72:00:00" >> ${OUT}
# echo "#PBS -l nodes=1:ppn=1:HIGHMEM" >> ${OUT}
# echo "#PBS -q highmem_q" >> ${OUT}
# echo "#PBS -l mem=200gb" >> ${OUT}
# echo "" >> ${OUT}
# echo "cd ${output_directory}" >> ${OUT}
# echo "module load ${GATK_module}" >> ${OUT}
# echo "" >> ${OUT}
# echo "time gatk HaplotypeCaller \
# -R ${ref_genome} \
# -ERC GVCF \
# -I ${output_directory}/bams/recalibrated/${BASE}_recalibratedNewRef.bam \
# -ploidy 2 \
# -O ${output_directory}/gVCFs/${BASE}_variants.Recal.g.vcf" >> ${OUT}
#
# qsub ${OUT}
#
# done


#
#           ###################################################################################################
#           ## Combine gvcfs
#           ###################################################################################################
#           ###################################################################################################
#           #
#           #
  # module load ${GATK_module}
#

# gatk ApplyBQSR \
#   -R ${ref_genome} \
#   -I ${output_directory}/D20-A__removedDuplicates.bam \
#   -bqsr ${output_directory}/D20-A__recal_data.table \
#   -O ${output_directory}/D20-A__recalibratedNewRef.bam

# time gatk HaplotypeCaller \
# -R ${ref_genome} \
# -ERC GVCF \
# -I ${output_directory}/D20-A__recalibratedNewRef.bam \
# -ploidy 2 \
# -O ${output_directory}/D20-A__variants.Recal.g.vcf

# interactively


# time gatk CombineGVCFs \
# -R ${ref_genome} \
# -O ${output_directory}/gVCFs/D20_FullCohort.g.vcf \
# -V ${output_directory}/gVCFs/D20_anc_switched/D20-A_R_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/D20-1_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-2_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-5_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-6_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-8_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-9_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-10_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-11_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-12_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-13_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-14_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-15_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-16_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-17_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-18_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-19_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-20_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-21_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-22_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-23_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-24_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-25_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-26_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-27_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-28_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-31_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-32_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-33_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-34_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-35_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-36_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-37_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-38_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-39_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-40_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-42_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-43_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/D20-44_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-45_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-46_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-47_variants.Recal.g.vcf \
# -V ${output_directory}/gVCFs/HM-D20-48_variants.Recal.g.vcf
# #
#              ###################################################################################################
#              ## Genotype gVCFs (jointly)
#              ###################################################################################################
#              ###################################################################################################
#
#
# time gatk GenotypeGVCFs \
#   -R ${ref_genome} \
#   -ploidy 2 \
#   --variant ${output_directory}/D20_FullCohort.g.vcf \
#   -O ${output_directory}/D20_FullCohort.vcf
#
# # ###################################################################################################
# # ### Find coverage and put into 10k chunks
# # ###################################################################################################
#
# module load ${bedtools_module}
# # report gives per-base depth across entire genome
#
# bedtools genomecov -d -ibam ${output_directory}/D20/D20-A__recalibratedNewRef.bam > ${output_directory}/D20/D20-A_depth.txt
#
#
#
#
#
# module load ${deeptools_module}
#
#
# for file in ${output_directory}/${BASE}*_recalibratedNewRef.bam
#
# do
#
# FBASE=$(basename $file _recalibratedNewRef.bam)
# BASE=${FBASE%_recalibratedNewRef.bam}
# OUT="${BASE}_bamCoverage.sh"
# echo "#!/bin/bash" > ${OUT}
# echo "#PBS -N ${BASE}_bamCoverage" >> ${OUT}
# echo "#PBS -l walltime=12:00:00" >> ${OUT}
# echo "#PBS -l nodes=1:ppn=1:AMD" >> ${OUT}
# echo "#PBS -q batch" >> ${OUT}
# echo "#PBS -l mem=20gb" >> ${OUT}
# echo "" >> ${OUT}
# echo "cd ${output_directory}" >> ${OUT}
# echo "module load ${deeptools_module}" >> ${OUT}
# echo "" >> ${OUT}
# echo "bamCoverage -b ${output_directory}/${BASE}_recalibratedNewRef.bam -o ${output_directory}/${BASE}.bedgraph -of bedgraph -bs 1" >> ${OUT}
# qsub ${OUT}
#
# done
#
#
# # for file in ${raw_data}/${BASE}*_piped.bam
# #
# # do
# #
# # FBASE=$(basename $file _piped.bam)
# # BASE=${FBASE%_piped.bam}
# # samtools sort ${raw_data}/${BASE}_piped.bam \
# # -o ${raw_data}/${BASE}.sorted.bam
# #
# # samtools depth \
# # ${raw_data}/${BASE}.sorted.bam \
# # |  awk '{sum+=$3} END { print "Average = ",sum/NR}' > ${raw_data}/${BASE}.txt
# #
# # done
# #
# # #combine depths with filenames into the same file
# # find . -type f -name "*.txt" -exec awk '{s=$0};END{if(s)print FILENAME,s}' {} \; > D0_depth.txt
#
#
# #                  # ################
# # # ###################################################################################################
# # # ### Filter variants
# # # Can easily run these interactively
# # # ###################################################################################################
# #
#
# ########################################################################
# #### Remove low and high read depth first
# gatk SelectVariants \
# -R ${ref_genome} \
# -V ${output_directory}/D20_FullCohort.vcf \
# -O ${output_directory}/D20_noLow.vcf \
# -select 'vc.getGenotype("D1-A_").getDP() > 82'
#
# gatk SelectVariants \
# -R ${ref_genome} \
# -V ${output_directory}/D20_noLow.vcf \
# -O ${output_directory}/D20_noLow_noHigh.vcf \
# -select 'vc.getGenotype("D1-A_").getDP() < 206'
#
# low_mappability="/scratch/jc33471/pilon/337/mappability/337_lowmappability.bed"
# module load ${bedtools_module}
#
# # bedtools sort -i ${low_mappability} > ${output_directory}/337_lowmappability_sorted.bed
# bedtools intersect -v -a ${output_directory}/D20_noLow_noHigh.vcf -b ${low_mappability} -header > ${output_directory}/D20_noLow_noHigh_redGem.vcf
#
# gatk SelectVariants \
# -R ${ref_genome} \
# -V ${output_directory}/D20_noLow_noHigh_redGem.vcf \
# -O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls.vcf \
# -select 'vc.getGenotype("D1-A_").isCalled()'
#
#
# gatk SelectVariants \
# -R ${ref_genome} \
# -V ${output_directory}/D20_noLow_noHigh_redGem_AncCalls.vcf \
# -O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets.vcf \
# -select '!vc.getGenotype("D1-A_").isHet()'
#
# ### Ancestor hets only
# gatk SelectVariants \
# -R ${ref_genome} \
# -V ${output_directory}/D20_noLow_noHigh_redGem_AncCalls.vcf \
# -O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_Hets.vcf \
# -select 'vc.getGenotype("D1-A_").isHet()'
#
#
# gatk SelectVariants \
#    -R ${ref_genome} \
#    -V ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets.vcf \
#    -O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets_SNPs.vcf \
#    --max-nocall-number 0 \
#    --exclude-non-variants TRUE \
# 	 --restrict-alleles-to BIALLELIC \
#    -select-type SNP
# #
# gatk SelectVariants \
#    -R ${ref_genome} \
#    -V ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets.vcf \
#    -O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets_Indels.vcf \
#    --max-nocall-number 0 \
# 	 --exclude-non-variants TRUE \
# 	 --restrict-alleles-to BIALLELIC \
#    -select-type INDEL
#
# gatk SelectVariants \
# 	 -R ${ref_genome} \
#    -V ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets.vcf \
#    -O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHetsVars.vcf \
#    --max-nocall-number 0 \
# 	 --exclude-non-variants TRUE \
# 	 --restrict-alleles-to BIALLELIC
#
#
# gatk VariantsToTable \
# 	 -V ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHetsVars.vcf \
# 	 -F CHROM -F POS -F REF -F ALT -F QUAL \
# 	 -GF AD -GF DP -GF GQ -GF GT \
# 	 -O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets_vars.txt
#
# gatk VariantsToTable \
# 	-V ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets_SNPs.vcf \
# 	-F CHROM -F POS -F REF -F ALT -F QUAL \
# 	-GF AD -GF DP -GF GQ -GF GT \
# 	-O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets_SNPs.txt
#
# gatk VariantsToTable \
# -V ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets_Indels.vcf \
# -F CHROM -F POS -F REF -F ALT -F QUAL \
# -GF AD -GF DP -GF GQ -GF GT \
# -O ${output_directory}/D20_noLow_noHigh_redGem_AncCalls_NoHets_Indels.txt
#
# #
# #    -select-type INDEL \
# #    -select-type MIXED \
# #    -select-type MNP \
# #    -select-type SYMBOLIC
# #
# # #
# # # #gives a final dataset with only called sites in the Ancestor, no heterozygous sites in the ancestor,
# # # # depth > 10, mapping quality > 50, and strand bias (SOR) > 0.01 (not significant)
# # #
# # # #Variants to table
# # gatk VariantsToTable \
# #      -V ${output_directory}/D20_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls.vcf \
# #      -F CHROM -F POS -F REF -F ALT -F QUAL \
# #      -GF AD -GF DP -GF GQ -GF GT \
# #      -O ${output_directory}/D20_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_vars.txt
# #
# #      gatk VariantsToTable \
# #           -V ${output_directory}/D20_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_SNPs.vcf \
# #           -F CHROM -F POS -F REF -F ALT -F QUAL \
# #           -GF AD -GF DP -GF GQ -GF GT \
# #           -O ${output_directory}/D20_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_SNPs.txt
# #
# #           gatk VariantsToTable \
# #                -V ${output_directory}/D20_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_Indels.vcf \
# #                -F CHROM -F POS -F REF -F ALT -F QUAL \
# #                -GF AD -GF DP -GF GQ -GF GT \
# #                -O ${output_directory}/D20_FullCohort_AnCalls_NoHets_DpGr10_MQGr50_StrBiasFil_Calls_Indels.txt
# #
# # #
# # #
# # gatk VariantsToTable \
# #      -V ${output_directory}/D20_FullCohort_AncCalls.vcf \
# #      -F CHROM -F POS -F REF -F ALT -F QUAL \
# #      -GF AD -GF DP -GF GQ -GF GT \
# #      -O ${output_directory}/D20_FullCohort_AncCalls.txt
# #
# #
# #
# #
# # ## VCF tools to extract all of the GT entries:
# #
# # vcftools --vcf file.vcf --extract-FORMAT-info GT
#
#
# samtools depth -a -d 100000 HM-D20-10_recalibratedNewRef.bam > HM-D20-10.depth
#
#
#
#
#
# module load HTSlib/1.6-foss-2016b
#
# cat H0_noLow.vcf | bgzip -c > H0_noLow.vcf.gz
# tabix H0_noLow.vcf.gz
#
# #            # ###################################################################################################
# #            # ### Find coverage and put into 10k chunks
# #            # ###################################################################################################
# #
# #            module load ${deeptools_module}
# #
# #
# #            for file in ${raw_data}/${BASE}*_piped.bam
# #
# #            do
# #
# #            FBASE=$(basename $file _piped.bam)
# #            BASE=${FBASE%_piped.bam}
# #
# #            bamCoverage -b ${raw_data}/${BASE}_piped.bam -o ${output_directory}/${BASE}.bedgraph -of bedgraph -bs 10000
# #
# #            done
# # for file in ${raw_data}/${BASE}*_piped.bam
# #
# # do
# #
# # FBASE=$(basename $file _piped.bam)
# # BASE=${FBASE%_piped.bam}
# # OUT="${BASE}_bamCoverage.sh"
# # echo "#!/bin/bash" >> ${OUT}
# # echo "#PBS -N ${BASE}_bamCoverage" >> ${OUT}
# # echo "#PBS -l walltime=12:00:00" >> ${OUT}
# # echo "#PBS -l nodes=1:ppn=1:AMD" >> ${OUT}
# # echo "#PBS -q batch" >> ${OUT}
# # echo "#PBS -l mem=20gb" >> ${OUT}
# # echo "" >> ${OUT}
# # echo "cd ${raw_data}" >> ${OUT}
# # echo "module load ${deeptools_module}" >> ${OUT}
# # echo "" >> ${OUT}
# # echo "bamCoverage -b ${raw_data}/${BASE}_piped.bam -o ${output_directory}/${BASE}.bedgraph -of bedgraph -bs 10000" >> ${OUT}
# # qsub ${OUT}
# #
# # done
#
# # for file in ${raw_data}/${BASE}*_piped.bam
# #
# # do
# #
# # FBASE=$(basename $file _piped.bam)
# # BASE=${FBASE%_piped.bam}
# # OUT="${BASE}_samtoolsDepth.sh"
# # echo "#!/bin/bash" > ${OUT}
# # echo "#PBS -N ${BASE}_samtoolsDepth" >> ${OUT}
# # echo "#PBS -l walltime=12:00:00" >> ${OUT}
# # echo "#PBS -l nodes=1:ppn=1:AMD" >> ${OUT}
# # echo "#PBS -q batch" >> ${OUT}
# # echo "#PBS -l mem=20gb" >> ${OUT}
# # echo "" >> ${OUT}
# # echo "cd ${raw_data}" >> ${OUT}
# # echo "module load ${samtools_module}" >> ${OUT}
# # echo "" >> ${OUT}
# #
# # echo "samtools sort ${raw_data}/${BASE}_piped.bam \
# # -o ${raw_data}/${BASE}.sorted.bam
# #
# # samtools depth \
# # ${raw_data}/${BASE}.sorted.bam \
# # |  awk '{sum+=$3} END { print "Average = ",sum/NR}' > ${raw_data}/${BASE}.txt" >> ${OUT}
# # qsub ${OUT}
# # done
# #
# # for file in ${raw_data}/${BASE}*_piped.bam
# #
# # do
# #
# # FBASE=$(basename $file _piped.bam)
# # BASE=${FBASE%_piped.bam}
# # samtools sort ${raw_data}/${BASE}_piped.bam \
# # -o ${raw_data}/${BASE}.sorted.bam
# #
# # samtools depth \
# # ${raw_data}/${BASE}.sorted.bam \
# # |  awk '{sum+=$3} END { print "Average = ",sum/NR}' > ${raw_data}/${BASE}.txt
# #
# # done
# # # ###################################################################################################
# # # ### Aggregate the GVCF files using GenomicsDBImport
# # # ###################################################################################################
# # # mkdir ${genomicsdb_workspace_path}
# # # mkdir ${tmp_DIR}
# # #
# # # gatk --java-options "-Xmx4g -Xms4g" \
# # #        GenomicsDBImport \
# # #        --genomicsdb-workspace-path ${genomicsdb_workspace_path} \
# # #        --batch-size 50 \
# # #        --sample-name-map ${sample_name_map} \
# # #        --TMP_DIR:${tmp_DIR} \
# # #        --reader-threads 12
