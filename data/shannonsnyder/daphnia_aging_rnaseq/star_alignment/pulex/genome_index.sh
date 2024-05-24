#!/bin/bash
                                                                    
#SBATCH -A bgmp                ### Account                                                                
#SBATCH --partition=memory   ### Partition                                                             
#SBATCH --job-name=star_index         ### Job Name                                                              
#SBATCH --time=03:00:00        ### WallTime                                                       
#SBATCH --nodes=1              ### Number of Nodes                                         
#SBATCH --ntasks=4             ### Number of tasks per array job
##SBATCH --array=0-3     
#SBATCH --mail-type=END,FAIL              ### Mail events (NONE, BEGIN, END, FA$                        \                               
#SBATCH --mail-user=your@gmail.com  ### Where to send mail                                              \
#SBATCH --cpus-per-task=12            ### Number of CPU cores per task 


# Load modules and dependencies
module load easybuild
module load icc/2017.1.132-GCC-6.3.0-2.27
module load impi/2017.1.132                                                                             \


#module load easybuild  icc/2017.1.132-GCC-6.3.0-2.27  impi/2017.1.132
       
module load STAR/2.5.3a


# Path to the STAR executable
#STAR="/path/to/star"

# Path to the genome FASTA file
genome_fasta="/projects/nereus/ssnyder3/aging_rnaseq/genomes/dpulex/ncbi-genomes-2023-05-22/GCF_021134715.1_A\
SM2113471v1_genomic.fna"

# Path to the directory where the index will be generated
index_dir="/projects/nereus/ssnyder3/aging_rnaseq/genomes/dpulex/ncbi-genomes-2023-05-22/indexDirectory_modload"

# Path to annotation file 
GTFfile_path="/projects/nereus/ssnyder3/aging_rnaseq/genomes/dpulex/ncbi-genomes-2023-05-22/GCF_021134715.1_ASM211347\
1v1_genomic.gtf" #note: I generated this file from a GFF using cufflinks

# Number of threads to use for indexing
num_threads=4

# Run STAR indexing
/usr/bin/time -v STAR --runMode genomeGenerate \
      --genomeDir $index_dir \
      --genomeFastaFiles $genome_fasta \
      --runThreadN $num_threads \
      --sjdbGTFfile $GTFfile_path \
      --genomeSAindexNbases 12

# Check if indexing was successful
if [ $? -eq 0 ]; then
  echo "Genome indexing completed successfully!"
else
  echo "Genome indexing failed!"
fi
