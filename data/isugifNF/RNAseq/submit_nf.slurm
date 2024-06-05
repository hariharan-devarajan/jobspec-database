#! /usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --job-name=NF_RNASeq
#SBATCH --output=R-%x.%J.out
#SBATCH --error=R-%x.%J.err
# --mail-user=username@email.com
# --mail-type=begin
# --mail-type=end
# --account=project_name    # <= put HPC account name here, required on Atlas

# === Load Modules here
# == Conda environment (replace with path to environment), might need 'module load miniconda'
# source activate ~/miniconda3/envs/rnaseq_env

# ==  Atlas HPC (will need a local install of nextflow)
# module load singularity
# NEXTFLOW=/project/isu_gif_vrsc/programs/nextflow

# == Ceres HPC
module load nextflow
NEXTFLOW=nextflow
# native Ceres modules (without using the conda environment.yml)
module load parallel fastqc python_3 salmon/0.10.1 kallisto/0.42.4 hisat2/2.2.0 gmap_gsnap/2020-04-08 samtools subread/2.0.2

# == Nova HPC
# module load gcc/7.3.0-xegsmw4 nextflow
# module load singularity
# NEXTFLOW=nextflow

# === Set working directory and in/out variables
cd ${SLURM_SUBMIT_DIR}

# === Main Program
${NEXTFLOW} run main.nf \
  --reads "00_Raw-Data/*{1,2}.fastq.gz" \
  --genome "00_Raw-Data/*_genomic.fna.gz" \
  --genome_gff "00_Raw-Data/*.gff.gz" \
  --genome_cdna "00_Raw-Data/*_rna.fna.gz" \
  --queueSize 25 \
  -profile slurm \
  -resume
  #--account isu_gif_vrsc       #<= add this to Atlas

