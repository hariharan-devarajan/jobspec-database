#!/bin/bash
#SBATCH --nodes=1 #request 1 node
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=04:00:00
#SBATCH --verbose
#SBATCH --job-name=align
#SBATCH --output=logs/%x.%j.out
#SBATCH --err=logs/%x.%j.err
#SBATCH --array=0-31 # please start array index at 0

# This scirpt will fail if you do not have a logs/ directory for
# the output and error files to go into.
# Change the following parameters: THREADS, BOWTIE2_IDX, and FASTQ_FILES. 
# also make sure the --array SBATCH goes from 0 to the number of samples - 1

#############
#CONFIG
##############

THREADS=8
BOWTIE_IDX="home/groups/MaxsonLab/indices/GRch38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.bowtie_index"
FASTQ_FILES="/home/groups/MaxsonLab/input-data/CUT_RUNTAG/2019/SETKAS_TAG_12_19/KAS_TAG_12_19/"

##############
#END CONFIG
###############

# activate conda env containing bowtie2 and samtools
# conda create -n align -y -c bioconda bowtie2 samtools
source activate align

# define arrays with read 1 and read2
r1s=(${FASTQ_FILES}*R1*)
r2s=(${r1s[@]//R1/R2})


# check that job array is zero indexed
if [[ "${SLURM_ARRAY_TASK_MIN}" -ne "0" ]]; then 
    echo "Please start your job array at and index of 0: --array=0-4 for five jobs."
    exit 1
fi

# check that you specify the same number of array jobs 
# as the number of samples to process
if [[ "${#r1s[@]}" -ne "${SLURM_ARRAY_TASK_COUNT}" ]]; then
    echo "You have ${#r1s[@]} sampls and only ${SLURM_ARRAY_TASK_MAX} array jobs.\
        Please specify the same number of array jobs as samples to process"
    exit 1
fi
        

# define read by reading from array 
# indexed by SLURM_ARR_TASK_ID 
R1=${r1s[$SLURM_ARRAY_TASK_ID]}
R2=${r2s[$SLURM_ARRAY_TASK_ID]}

# exit if they are not
if [[ "${R1%%_R1*}" != "${R2%%_R2*}" ]]; then 
    echo "Sample names do not match for alignment!"
    echo "${R1%%_R1*} != ${R2%%_R2*}"
    exit 1
else
    # name the sample using part of string before _R1
    # that we just checked for equality with it's mate
    tmp=$(basename ${r1s[$SLURM_ARRAY_TASK_ID]})
    SAMPLE=${tmp%%_R1*}
fi

OUTPUT="data/bams/${SAMPLE}.bam"
LOG="data/bams/${SAMPLE}.bowtie2.log
echo "${OUTPUT}"

# align PE reads and
echo "bowtie2 --local --very-sensitive-local --no-unal --no-mixed \
    --threads ${THREADS} --no-discordant --phred33 \
    -I 10 -X 700 -x ${BOWTIE_IDX} -1 ${R1} -2 ${R2} \
    2>${LOG} | samtools view -Sbh - > ${OUTPUT}"
