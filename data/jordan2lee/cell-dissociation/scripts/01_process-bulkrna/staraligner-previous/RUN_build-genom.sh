#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=docker-buildgenom
#SBATCH --time=0-04:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --output=../../logs/slurm.%N.%j.out
#SBATCH --error=../../logs/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leejor@ohsu.edu


dir=/home/groups/EllrottLab/cell-dissociation
ses=build-ref-genom
threads=11

################
# Set up workspace
################
# Move to main dir
cd ${dir}
# make output structure (for this script and RUN.sh)
mkdir /mnt/scratch/5420
mkdir /mnt/scratch/5420/${ses}
mkdir /mnt/scratch/5420/${ses}/input
mkdir /mnt/scratch/5420/${ses}/input/reference_files
mkdir /mnt/scratch/5420/${ses}/output
mkdir /mnt/scratch/5420/${ses}/output/genome_ref

# if not already there, get metadata + unzip for STAR
cd /mnt/scratch/5420/${ses}/input/reference_files
# Human FA for building ref genome
wget ftp://ftp.ensembl.org/pub/release-98/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
zcat /mnt/scratch/5420/${ses}/input/reference_files/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz > /mnt/scratch/5420/${ses}/input/TEMP-ref.fa
# Human GTF for building ref genome (and used in RUN.sh)
wget ftp://ftp.ensembl.org/pub/release-98/gtf/homo_sapiens/Homo_sapiens.GRCh38.98.gtf.gz
zcat /mnt/scratch/5420/${ses}/input/reference_files/Homo_sapiens.GRCh38.98.gtf.gz > /mnt/scratch/5420/${ses}/input/TEMP-ref.gtf

# Move to main dir
cd ${dir}

##########################
# RUN ONLY ONCE
# Build genome indexes for alignment
#       genomeSAsparseD=1 (default)
##########################
echo '##### Building genome indexes #####'
sudo /opt/acc/sbin/exadocker pull alexdobin/star:2.6.1d
sudo /opt/acc/sbin/exadocker run --rm -v /mnt/scratch/5420/${ses}:/tmp \
    alexdobin/star:2.6.1d \
    STAR \
    --runThreadN ${threads} \
    --runMode genomeGenerate \
    --genomeDir /tmp/output/genome_ref/ \
    --genomeFastaFiles /tmp/input/TEMP-ref.fa \
    --sjdbGTFfile /tmp/input/TEMP-ref.gtf \
    --sjdbOverhang 100 \
    --outFileNamePrefix /tmp/output/genome_ref/Log \
    --genomeSAsparseD 1


###############
# Clean up workspace
###############
# copy files from scratch and clean up scratch
echo '##### Cleaning up workspace #####'
cp -r /mnt/scratch/5420/${ses}/output/* /home/groups/EllrottLab/cell-dissociation/raw-data
cp -r /mnt/scratch/5420/${ses}/input/reference_files /home/groups/EllrottLab/cell-dissociation/raw-data
rm -rf /mnt/scratch/5420/${ses}
