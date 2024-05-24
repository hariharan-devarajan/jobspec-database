#!/bin/bash
## Job Name
#SBATCH --job-name=pgen_v074_stringtie
## Allocation Definition
#SBATCH --account=srlab
#SBATCH --partition=srlab
## Resources
## Nodes
#SBATCH --nodes=1
## Walltime (days-hours:minutes:seconds format)
#SBATCH --time=25-00:00:00
## Memory per node
#SBATCH --mem=500G
##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=samwhite@uw.edu
## Specify the working directory for this job
#SBATCH --workdir=/gscratch/scrubbed/samwhite/outputs/20190723_stringtie_pgen_v074

# Exit script if any command fails
set -e

# Load Python Mox module for Python module availability

module load intel-python3_2017

# Document programs in PATH (primarily for program version ID)

date >> system_path.log
echo "" >> system_path.log
echo "System PATH for $SLURM_JOB_ID" >> system_path.log
echo "" >> system_path.log
printf "%0.s-" {1..10} >> system_path.log
echo "${PATH}" | tr : \\n >> system_path.log


threads=28
genome_index_name="Pgenerosa_v074"

# Paths to programs
hisat2_dir="/gscratch/srlab/programs/hisat2-2.1.0"
hisat2="${hisat2_dir}/hisat2"
samtools="/gscratch/srlab/programs/samtools-1.9/samtools"
stringtie="/gscratch/srlab/programs/stringtie-1.3.6.Linux_x86_64/stringtie"

# Input/output files
genome_gff="/gscratch/srlab/sam/data/P_generosa/genomes/Pgenerosa_v074_genome_snap02.all.renamed.putative_function.domain_added.gff"
genome_index_dir="/gscratch/srlab/sam/data/P_generosa/genomes"
fastq_dir="/gscratch/scrubbed/samwhite/data/P_generosa/RNAseq/"
gtf_list="gtf_list.txt"

## Inititalize arrays
fastq_array_R1=()
fastq_array_R2=()
names_array=()

# Copy Hisat2 genome index
rsync -av "${genome_index_dir}"/Pgenerosa_v074*.ht2 .

# Create array of fastq R1 files
for fastq in "${fastq_dir}"*R1*.gz
do
  fastq_array_R1+=("${fastq}")
done

# Create array of fastq R2 files
for fastq in "${fastq_dir}"*R2*.gz
do
  fastq_array_R2+=("${fastq}")
done

# Create array of sample names
## Uses parameter substitution to strip leading path from filename
## Uses awk to parse out sample name from filename
for R1_fastq in "${fastq_dir}"*R1*.gz
do
  names_array+=($(echo "${R1_fastq#${fastq_dir}}" | awk -F"_" '{print $1}'))
done

# Create list of fastq files used in analysis
## Uses parameter substitution to strip leading path from filename
for fastq in "${fastq_dir}"*.gz
do
  echo "${fastq#${fastq_dir}}" >> fastq.list.txt
done

# Hisat2 alignments
for index in "${!fastq_array_R1[@]}"
do
  sample_name=$(echo "${names_array[index]}")
  "${hisat2}" \
  -x "${genome_index_name}" \
  --downstream-transcriptome-assembly \
  -1 "${fastq_array_R1[index]}" \
  -2 "${fastq_array_R2[index]}" \
  -S "${sample_name}".sam \
  2> "${sample_name}"_hisat2.err

# Sort SAM files, convert to BAM, and index
  "${samtools}" view \
  -@ "${threads}" \
  -Su "${sample_name}".sam \
  | "${samtools}" sort - \
  -@ "${threads}" \
  -o "${sample_name}".sorted.bam
  "${samtools}" index "${sample_name}".sorted.bam

# Run stringtie on alignments
  "${stringtie}" "${sample_name}".sorted.bam \
  -p "${threads}" \
  -o "${sample_name}".gtf \
  -G "${genome_gff}" \
  -C "${sample_name}.cov_refs.gtf"

# Add GTFs to list file, only if non-empty
# Identifies GTF files that only have header
  gtf_lines=$(wc -l < ${sample_name}.gtf )
  if [ "${gtf_lines}" -gt 2 ]; then
    echo "${sample_name}.gtf" >> "${gtf_list}"
  fi
done

# Create singular transcript file, using GTF list file
"${stringtie}" --merge \
"${gtf_list}" \
-p "${threads}" \
-G "${genome_gff}" \
-o "${genome_index_name}".stringtie.gtf

# Delete unneccessary index files
rm "${genome_index_name}"*.ht2

# Delete unneded SAM files
rm ./*.sam
