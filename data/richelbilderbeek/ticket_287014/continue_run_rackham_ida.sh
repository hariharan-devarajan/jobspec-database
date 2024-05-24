#!/bin/bash
#SBATCH -J metontiime_27mar
#SBATCH -A naiss2023-22-866
#SBATCH -t 9-00:00:00
#SBATCH -p core
#SBATCH -n 4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ida.nordstrom@slu.se
#SBATCH --error=metontiime_27mar.err
#SBATCH --output=metontiime_27mar.out

#
# Resumes the run
#
module load bioinfo-tools Nextflow 



# work_dir="${PWD}/work_local_singularity"
# work_dir="/crex/proj/staff/richel/ticket_287014_output/work_local_singularity"
work_dir="/crex/proj/naiss2023-22-866/MetONTIIME/trimmed_and_filtered_Q15_qz"
echo "work_dir: ${work_dir}"

# No, these are already there :-)
# # Get the sequences in the correct folder 
# mkdir "${work_dir}"
# # cp example_reads.fastq.gz "${work_dir}"
# cp /crex/proj/naiss2023-22-866/MetONTIIME/trimmed_and_filtered_Q15_qz/*.fastq.gz "${work_dir}"

# results_dir="${PWD}/results_local_singularity"
# results_dir="/crex/proj/staff/richel/ticket_287014_output/results_local_singularity"
results_dir="/crex/proj/naiss2023-22-866/MetONTIIME/Output"
echo "results_dir: ${results_dir}"

# db_sequence_fasta_filename="${PWD}/example_db_sequence.fasta"
# db_sequence_fasta_filename="/crex/proj/naiss2023-22-866/MetONTIIME/sh_refs_qiime_ver9_dynamic_all_25.07.2023.fasta"
db_sequence_fasta_filename="/crex/proj/naiss2023-22-866/Illumina_Qiime2_and_BLAST/sh_refs_qiime_ver9_dynamic_all_25.07.2023.fasta"
echo "db_sequence_fasta_filename: ${db_sequence_fasta_filename}"

# # From Simone Maestri:
# # > the sample metadata file is optional, 
# # > you don't need to create one, you can specify the path to an empty file, 
# # > which will be created upon running
# From Richel: you do need to specify the path-to_be
sample_metadata_tsv_filename="${PWD}/example_sample_metadata.tsv"
echo "sample_metadata_tsv_filename: ${sample_metadata_tsv_filename}"

#taxonomy_tsv_filename="${PWD}/example_taxonomy.tsv"
taxonomy_tsv_filename="/crex/proj/naiss2023-22-866/MetONTIIME/noheader_taxonomy_qiime_ver9_dynamic_alleukaryotes_25.07.2023.tsv"
echo "taxonomy_tsv_filename: ${taxonomy_tsv_filename}"

config_filename="${PWD}/metontiime2.conf"

# Resume and run locally
#
# The only new thing is the '-resume' flag
nextflow -c "${config_filename}" \
  run -resume metontiime2.nf \
  --workDir="${work_dir}" \
  --resultsDir="${results_dir}" \
  --dbSequencesFasta="${db_sequence_fasta_filename}" \
  --sampleMetadata="${sample_metadata_tsv_filename}" \
  --dbTaxonomyTsv="${taxonomy_tsv_filename}" \
  -profile singularity
