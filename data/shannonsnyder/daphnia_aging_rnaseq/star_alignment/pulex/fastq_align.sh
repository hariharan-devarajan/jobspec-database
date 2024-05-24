#!/bin/bash

#SBATCH -A bgmp       ### Account                                                                                 
#SBATCH --partition=memory   ### Partition                                                                           
#SBATCH --job-name=star_align         ### Job Name                                                                   
#SBATCH --time=18:00:00        ### WallTime                                                                          
#SBATCH --nodes=3              ### Number of Nodes                                                                   
#SBATCH --ntasks=1             ### Number of tasks per array job                                                     
##SBATCH --array=0-3           ### Array index                                                                       
#SBATCH --mail-type=END,FAIL              ### Mail events (NONE, BEGIN, END, FA$                                     
##SBATCH --mail-user=ssnyder3@uoregon.edu  ### Where to send mail                                                     
#SBATCH --cpus-per-task=12            ### Number of CPU cores per task                                                

#load modules                                                                                                        
module load easybuild
module load icc/2017.1.132-GCC-6.3.0-2.27
module load impi/2017.1.132
module load STAR/2.5.3a

# Set the paths to the indexed reference genome
REFERENCE_GENOME="/home/ssnyder3/nereus/aging_rnaseq/genomes/dpulex/ncbi-genomes-2023-05-22/indexDirectory_modload"

# Set the input and output directories
FASTQ_DIR="/home/ssnyder3/nereus/aging_rnaseq/raw_fastq/pulex_rawfastqs"
OUTPUT_DIR="/home/ssnyder3/nereus/aging_rnaseq/star_alignment/pulex/starAlign_output_modload"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over the paired-end FASTQ files in the input directory
#for file in "$FASTQ_DIR"/*.fastq.gz; do
  # Extract the base filename without the extension
#  base=$(basename "$file" .fastq.gz)

# Loop through the paired-end FASTQ files
for fileR1 in $FASTQ_DIR/*R1_001.fastq.gz; do
    # Extract the file name without extension
    filename=$(basename "$fileR1" _R1_001.fastq.gz)
    echo $fileR1
    # Determine the corresponding R2 file
    fileR2="${fileR1/_R1/_R2}" #newvariablename=${oldvariablename//oldtext/newtext}
    
    # Run STAR alignment
    STAR --runThreadN 4 \
          --genomeDir $REFERENCE_GENOME \
          --readFilesIn $fileR1 $fileR2 \
	  --quantMode GeneCounts \
	  --readFilesCommand zcat \
	  --outSAMtype BAM SortedByCoordinate \
          --outFileNamePrefix $OUTPUT_DIR/$filename"_"
done



