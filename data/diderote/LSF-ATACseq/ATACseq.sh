#!/bin/bash
#BSUB -J ATACseq-pipe
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q bigmem
#BSUB -R 'rusage[mem=110000]'
#BSUB -n 1
#BSUB -W 120:00

# YAML TEMPLATE ATACseq-experiment-file.yml

module rm python
module rm R
source activate chipseq-pipeline

#parsing replicate number from yaml file
REPS=$(cat $1 | yq -r .replicate_number)

if [ ${REPS} = '1' ]; then

	#parsing variables from yaml file
	TITLE=$(cat $1 | yq -r .title)
	RUNDATE=$(cat $1 | yq -r .rundate)
	SYSTEM=$(cat $1 | yq -r .system)
	GENOME=$(cat $1 | yq -r .genome)
	FASTQ1=$(cat $1 | yq -r .file_names.fastq1)
	FQDIR=$(cat $1 | yq -r .directories.fastq_directory)
	OUTDIR=$(cat $1 | yq -r .directories.output_directory)
	SCRATCH=$(cat $1 | yq -r .directories.scratch_directory)
	#SUBSAMPLE=$(cat $1 | yq -r .subsample) #NOT IN USE IN THIS PIPE
	#SUBNUM=$(cat $1 | yq -r .subsample_number) #NOT IN USE IN THIS PIPE 

	echo "This file was processed using ATACseq-ENCODE3-hg38-2017-08-17.sh"
	echo "Daniel Karl - Nimer Lab, Sylvester Comprehensive Cancer Center, University of Miami"
	echo "TITLE = ${TITLE}"
	echo "RUNDATE = ${RUNDATE}"
	echo "SYSTEM = ${SYSTEM}"
	echo "GENOME = ${GENOME}"
	echo "FASTQ1 = ${FASTQ1}"
	echo "FASTQ2 = ${FASTQ2}"
	echo "FASTQ3 = ${FASTQ3}"
	echo "FQDIR = ${FQDIR}"
	echo "OUTDIR = ${OUTDIR}"
	echo "SCRATCH = ${SCRATCH}"
	echo "Processing ONE sample."
	#echo "SUBSAMPLE = ${SUBSAMPLE}"
	#echo "SUBNUM = ${SUBNUM}"

	mkdir -p ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ1}_R1.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ1}_R2.fastq.gz ${SCRATCH}/${TITLE}/

	bds /projects/ctsi/nimerlab/DANIEL/tools/atac_dnase_pipelines/atac.bds \
		-out_dir ${SCRATCH}/${TITLE} \
		-title ${TITLE} \
		-system ${SYSTEM} \
		-species ${GENOME} \
		-ENCODE3 \
		-unlimited_mem_wt true \
		-enable_idr true \
		-nth 15 \
		-rm_chr_from_tag mito \
		-fastq1_1 ${SCRATCH}/${TITLE}/${FASTQ1}_R1.fastq.gz -fastq1_2 ${SCRATCH}/${TITLE}/${FASTQ1}_R2.fastq.gz \
		
	mkdir -p ${OUTDIR}/${TITLE}

	rm ${SCRATCH}/${TITLE}/*.gz
	cp -r ${SCRATCH}/${TITLE} ${OUTDIR}/${TITLE}

elif [ ${REPS} = '2' ]; then
	
	#parsing variables from yaml file
	TITLE=$(cat $1 | yq -r .title)
	RUNDATE=$(cat $1 | yq -r .rundate)
	SYSTEM=$(cat $1 | yq -r .system)
	GENOME=$(cat $1 | yq -r .genome)
	FASTQ1=$(cat $1 | yq -r .file_names.fastq1)
	FASTQ2=$(cat $1 | yq -r .file_names.fastq2)
	FQDIR=$(cat $1 | yq -r .directories.fastq_directory)
	OUTDIR=$(cat $1 | yq -r .directories.output_directory)
	SCRATCH=$(cat $1 | yq -r .directories.scratch_directory)
	#SUBSAMPLE=$(cat $1 | yq -r .subsample) #NOT IN USE IN THIS PIPE
	#SUBNUM=$(cat $1 | yq -r .subsample_number) #NOT IN USE IN THIS PIPE 

	echo "This file was processed using ATACseq-ENCODE3-hg38-2017-08-17.sh"
	echo "Daniel Karl - Nimer Lab, Sylvester Comprehensive Cancer Center, University of Miami"
	echo "TITLE = ${TITLE}"
	echo "RUNDATE = ${RUNDATE}"
	echo "SYSTEM = ${SYSTEM}"
	echo "GENOME = ${GENOME}"
	echo "FASTQ1 = ${FASTQ1}"
	echo "FASTQ2 = ${FASTQ2}"
	echo "FQDIR = ${FQDIR}"
	echo "OUTDIR = ${OUTDIR}"
	echo "SCRATCH = ${SCRATCH}"
	echo "Processing TWO Replicates."
	#echo "SUBSAMPLE = ${SUBSAMPLE}"
	#echo "SUBNUM = ${SUBNUM}"

	mkdir -p ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ1}_R1.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ1}_R2.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ2}_R1.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ2}_R2.fastq.gz ${SCRATCH}/${TITLE}/

	bds /projects/ctsi/nimerlab/DANIEL/tools/atac_dnase_pipelines/atac.bds \
		-out_dir ${SCRATCH}/${TITLE} \
		-title ${TITLE} \
		-unlimited_mem_wt true \
		-system ${SYSTEM} \
		-species ${GENOME} \
		-ENCODE3 \
		-enable_idr true \
		-nth 15 \
		-rm_chr_from_tag mito \
		-fastq1_1 ${SCRATCH}/${TITLE}/${FASTQ1}_R1.fastq.gz -fastq1_2 ${SCRATCH}/${TITLE}/${FASTQ1}_R2.fastq.gz \
		-fastq2_1 ${SCRATCH}/${TITLE}/${FASTQ2}_R1.fastq.gz -fastq2_2 ${SCRATCH}/${TITLE}/${FASTQ2}_R2.fastq.gz \

	mkdir -p ${OUTDIR}/${TITLE}

	rm ${SCRATCH}/${TITLE}/*.gz
	cp -r ${SCRATCH}/${TITLE} ${OUTDIR}/${TITLE}

elif [ ${REPS} = '3' ]; then

	#parsing variables from yaml file
	TITLE=$(cat $1 | yq -r .title)
	RUNDATE=$(cat $1 | yq -r .rundate)
	SYSTEM=$(cat $1 | yq -r .system)
	GENOME=$(cat $1 | yq -r .genome)
	FASTQ1=$(cat $1 | yq -r .file_names.fastq1)
	FASTQ2=$(cat $1 | yq -r .file_names.fastq2)
	FASTQ3=$(cat $1 | yq -r .file_names.fastq3)
	FQDIR=$(cat $1 | yq -r .directories.fastq_directory)
	OUTDIR=$(cat $1 | yq -r .directories.output_directory)
	SCRATCH=$(cat $1 | yq -r .directories.scratch_directory)
	REPS=$(cat $1 | yq -r .replicate_number)
	SUBSAMPLE=$(cat $1 | yq -r .subsample) #NOT IN USE IN THIS PIPE
	SUBNUM=$(cat $1 | yq -r .subsample_number) #NOT IN USE IN THIS PIPE

	echo "This file was processed using ATACseq-ENCODE3-hg38-2017-08-17.sh"
	echo "Daniel Karl - Nimer Lab, Sylvester Comprehensive Cancer Center, University of Miami"
	echo "TITLE = ${TITLE}"
	echo "RUNDATE = ${RUNDATE}"
	echo "SYSTEM = ${SYSTEM}"
	echo "GENOME = ${GENOME}"
	echo "FASTQ1 = ${FASTQ1}"
	echo "FASTQ2 = ${FASTQ2}"
	echo "FASTQ3 = ${FASTQ3}"
	echo "FQDIR = ${FQDIR}"
	echo "OUTDIR = ${OUTDIR}"
	echo "SCRATCH = ${SCRATCH}"
	echo "Processing THREE Replicates."


	mkdir -p ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ1}_R1.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ1}_R2.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ2}_R1.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ2}_R2.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ3}_R1.fastq.gz ${SCRATCH}/${TITLE}/
	cp ${FQDIR}/${FASTQ3}_R2.fastq.gz ${SCRATCH}/${TITLE}/

	bds /projects/ctsi/nimerlab/DANIEL/tools/atac_dnase_pipelines/atac.bds \
		-out_dir ${SCRATCH}/${TITLE} \
		-unlimited_mem_wt true \
		-title ${TITLE} \
		-system ${SYSTEM} \
		-species ${GENOME} \
		-ENCODE3 \
		-enable_idr true \
		-nth 15 \
		-rm_chr_from_tag mito \
		-fastq1_1 ${SCRATCH}/${TITLE}/${FASTQ1}_R1.fastq.gz -fastq1_2 ${SCRATCH}/${TITLE}/${FASTQ1}_R2.fastq.gz \
		-fastq2_1 ${SCRATCH}/${TITLE}/${FASTQ2}_R1.fastq.gz -fastq2_2 ${SCRATCH}/${TITLE}/${FASTQ2}_R2.fastq.gz \
		-fastq3_1 ${SCRATCH}/${TITLE}/${FASTQ3}_R1.fastq.gz -fastq3_2 ${SCRATCH}/${TITLE}/${FASTQ3}_R2.fastq.gz \

	mkdir -p ${OUTDIR}/${TITLE}

	rm ${SCRATCH}/${TITLE}/*.gz
	cp -r ${SCRATCH}/${TITLE} ${OUTDIR}/${TITLE}

else

	echo "This pipeline processes one, two, or three biological replicates."

fi
