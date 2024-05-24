#!/bin/bash

#SBATCH --job-name=Trinity
#SBATCH -o trin_%j.out
#SBATCH -e trin_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email@school.edu
#SBATCH --ntasks 16
#SBATCH --mem-per-cpu=4gb
#SBATCH -t 24:00:00
#SBATCH -p intel

date

#unload older version of perl
module unload perl/5.20.2

#load needed modules
module load perl/5.22.0
module load trinity-rnaseq/2.4.0
module load samtools/1.4.1
module load bowtie/1.1.1
module load rsem/1.3.0
module load transdecoder/3.0.0
module load bowtie2/2.2.9


#Trinity help
#Trinity --show_full_usage_info

#assemble a denovo transcript using all possible reads
Trinity --seqType fq --single /rhome/jlandis/bigdata/Nicotiana/Seq_reads/de_novo/cleaned_files2/Ntab51789_reads.fastq,/rhome/jlandis/bigdata/Nicotiana/Seq_reads/de_novo/cleaned_files2/Ntab095_55_reads.fastq --max_memory 127G --CPU 8 --min_contig_length 250 --output /rhome/jlandis/bigdata/Nicotiana/de_novo_approach/trinity_Ntab --full_cleanup

#prep resulting fasta file for mapping (needs to be done regardless of what reads want to map for abundance)
/opt/linux/centos/7.x/x86_64/pkgs/trinity-rnaseq/2.4.0/util/align_and_estimate_abundance.pl --transcripts Nicotiana_insilico.Trinity.fasta --est_method RSEM --aln_method bowtie --trinity_mode --prep_reference

#map all reads back to the reference, this would generate an idea of contigs that are "real"
/opt/linux/centos/7.x/x86_64/pkgs/trinity-rnaseq/2.4.0/util/align_and_estimate_abundance.pl --transcripts Nicotiana_insilico.Trinity.fasta --seqType fq --single /rhome/jlandis/bigdata/Nicotiana/Seq_reads/de_novo/cleaned_files2/Ntab51789_reads.fastq,/rhome/jlandis/bigdata/Nicotiana/Seq_reads/de_novo/cleaned_files2/Ntab095_55_reads.fastq,/rhome/jlandis/bigdata/Nicotiana/Seq_reads/New_reads/All_QM_reads.fastq,/rhome/jlandis/bigdata/Nicotiana/Seq_reads/New_reads/All_sylv_reads.fastq,/rhome/jlandis/bigdata/Nicotiana/Seq_reads/New_reads/All_tab_reads.fastq,/rhome/jlandis/bigdata/Nicotiana/Seq_reads/New_reads/All_tom_reads.fastq --est_method RSEM --aln_method bowtie --trinity_mode --thread_count 16 --output_prefix Nicotiana --output_dir mapping/

#Transdecoder, translate reference transcript into necessary components for downstream analysis and annotating
TransDecoder.LongOrfs -t Nicotiana_insilico.Trinity.fasta --gene_trans_map Nicotiana_insilico.Trinity.fasta.gene_trans_map

#Map individual reads back to reference, using biological replicate information
/opt/linux/centos/7.x/x86_64/pkgs/trinity-rnaseq/2.4.0/util/align_and_estimate_abundance.pl --transcripts /rhome/jlandis/bigdata/Nicotiana/de_novo_approach/In_silico_assembly/Nicotiana_insilico.Trinity.fasta --seqType fq --samples_file Samples.txt --est_method RSEM --aln_method bowtie2 --trinity_mode --thread_count 16

#generate expression matrix
#need to include a file (target_files.txt) to tell where the genes.results files are located
/opt/linux/centos/7.x/x86_64/pkgs/trinity-rnaseq/2.4.0/util/abundance_estimates_to_matrix.pl --est_method RSEM --quant_files target_files.txt --gene_trans_map /rhome/jlandis/bigdata/Nicotiana/de_novo_approach/In_silico_assembly/Nicotiana_insilico.Trinity.fasta.gene_trans_map --name_sample_by_basedir

#perform DE analysis with voom
mkdir DE_voom
/opt/linux/centos/7.x/x86_64/pkgs/trinity-rnaseq/2.4.0/Analysis/DifferentialExpression/run_DE_analysis.pl  --matrix matrix.counts.matrix --method voom --samples_file DE_samples.txt --output DE_voom
