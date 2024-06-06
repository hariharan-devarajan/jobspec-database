#!/bin/bash                                                                                                                                             
#SBATCH -A bgmp       ### Account                                                                                                                       
#SBATCH --partition=memory   ### Partition                                                                        \                                     

#SBATCH --job-name=sort_bams         ### Job Name                                                                                                       
#SBATCH --time=24:00:00        ### WallTime                                                                                                             

#SBATCH --nodes=1              ### Number of Nodes                                                                                                      
#SBATCH --ntasks=1             ### Number of tasks per array job                                                                                        
##SBATCH --array=0-3           ### Array index                                                                                                          
#SBATCH --mail-type=END,FAIL              ### Mail events (NONE, BEGIN, END, FA$                                                                        
##SBATCH --mail-user=ssnyder3@uoregon.edu  ### Where to send mail                                                 #SBATCH --cpus-per-task=12           \
 ### Number of CPU cores per task                                                                                                                       


# Input directory containing BAM files                                                                                                                  
input_dir="/home/ssnyder3/nereus/aging_rnaseq/star_alignment/pulex/aligned_reads"


# Output directory for sorted BAM files                                                                                                                 
mkdir -p sorted_bams
output_dir="/home/ssnyder3/nereus/aging_rnaseq/star_alignment/pulex/aligned_reads/sorted_bams"

# Loop through BAM files in the input directory                                                                                                         
for bam_file in "$input_dir"/*.bam; do
    # Get the base name of the BAM file                                                                                                                 
    base_name=$(basename "$bam_file" .bam)

    # Define the output sorted BAM file name                                                                                                            
    sorted_bam="${output_dir}/${base_name}_sorted.bam"

    # Use samtools to sort the BAM file                                                                                                                 
    samtools sort -o "$sorted_bam" "$bam_file"

    echo "Sorted BAM file created: $sorted_bam"
done

echo "All BAM files sorted and renamed."
