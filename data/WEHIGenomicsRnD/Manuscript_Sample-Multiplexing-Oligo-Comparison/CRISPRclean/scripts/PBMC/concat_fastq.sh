#!/bin/bash

#SBATCH --job-name=downsample
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -e ds-%j.err
#SBATCH -o ds-%j.out

# Downsample to the same number of reads to allow a fair comparison of Jumpcode CRISPRclean with untreated

module load gcc
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate ~/scratchHOME/ngsQC

# Number of reads to downsample to
DOWNSAMPLE=199000000;

# For Jumpcode PBMC I need to combine files prior to downsampling
# Need to concatnate them together into 1 fastq file again

INPUT_DIR="/stornext/Projects/score/GenomicsRnD/DB/NN265/fastq/AAAL25YHV";
OUTPUT_DIR="/stornext/Projects/score/GenomicsRnD/DB/NN265/fastq/AAAL25YHV/concat/";

# cat R010_LMO_JPC_GEX_S1_L00*_I1_001.fastq.gz > concat/R010_LMO_JPC_GEX_S1_L001_I1_001.fastq.gz;
# cat R010_LMO_JPC_GEX_S1_L00*_R1_001.fastq.gz > concat/R010_LMO_JPC_GEX_S1_L001_R1_001.fastq.gz;
# cat R010_LMO_JPC_GEX_S1_L00*_R2_001.fastq.gz > concat/R010_LMO_JPC_GEX_S1_L001_R2_001.fastq.gz;
# 
# cat R010_LMO_UTD_GEX_S5_L00*_I1_001.fastq.gz > concat/R010_LMO_UTD_GEX_S5_L001_I1_001.fastq.gz;
# cat R010_LMO_UTD_GEX_S5_L00*_R1_001.fastq.gz > concat/R010_LMO_UTD_GEX_S5_L001_R1_001.fastq.gz;
# cat R010_LMO_UTD_GEX_S5_L00*_R2_001.fastq.gz > concat/R010_LMO_UTD_GEX_S5_L001_R2_001.fastq.gz;

# Downsampling

seqtk sample -2 -s100 concat/R010_LMO_JPC_GEX_S1_L001_I1_001.fastq.gz $DOWNSAMPLE | gzip >  downsample/R010_LMO_JPC_GEX_S1_L001_I1_001.fastq.gz;
seqtk sample -2 -s100 concat/R010_LMO_JPC_GEX_S1_L001_R1_001.fastq.gz $DOWNSAMPLE | gzip >  downsample/R010_LMO_JPC_GEX_S1_L001_R1_001.fastq.gz;
seqtk sample -2 -s100 concat/R010_LMO_JPC_GEX_S1_L001_R2_001.fastq.gz $DOWNSAMPLE | gzip >  downsample/R010_LMO_JPC_GEX_S1_L001_R2_001.fastq.gz;

seqtk sample -2 -s100 concat/R010_LMO_UTD_GEX_S5_L001_I1_001.fastq.gz $DOWNSAMPLE | gzip >  downsample/R010_LMO_UTD_GEX_S5_L001_I1_001.fastq.gz;
seqtk sample -2 -s100 concat/R010_LMO_UTD_GEX_S5_L001_R1_001.fastq.gz $DOWNSAMPLE | gzip >  downsample/R010_LMO_UTD_GEX_S5_L001_R1_001.fastq.gz;
seqtk sample -2 -s100 concat/R010_LMO_UTD_GEX_S5_L001_R2_001.fastq.gz $DOWNSAMPLE | gzip >  downsample/R010_LMO_UTD_GEX_S5_L001_R2_001.fastq.gz;

# Combine prior to use with zUMI

cat concat/R010_LMO_JPC_GEX_S1_L001_I1_001.fastq.gz concat/R010_LMO_UTD_GEX_S5_L001_I1_001.fastq.gz \
    > concat/R010_PBMC_UTD-JPC_LMO_GEX_I1.fastq.gz;
    
cat concat/R010_LMO_JPC_GEX_S1_L001_R1_001.fastq.gz concat/R010_LMO_UTD_GEX_S5_L001_R1_001.fastq.gz \
    > concat/R010_PBMC_UTD-JPC_LMO_GEX_R1.fastq.gz;
    
cat concat/R010_LMO_JPC_GEX_S1_L001_R2_001.fastq.gz concat/R010_LMO_UTD_GEX_S5_L001_R2_001.fastq.gz \
    > concat/R010_PBMC_UTD-JPC_LMO_GEX_R2.fastq.gz;
    