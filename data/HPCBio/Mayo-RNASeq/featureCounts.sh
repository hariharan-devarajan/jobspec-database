#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 8G
#SBATCH --job-name=counts
#SBATCH --array=1-4
#SBATCH -p classroom

#Change directory to the main one containing data/, results/ and src/
cd ~/mouse-rnaseq-2020/

#Load the Subread module; name may change in future so check each time with 'module avail' on command line
module load Subread/2.0.0-IGB-gcc-8.2.0

#Set up variable to pull out base file names
line=$(sed -n -e "$SLURM_ARRAY_TASK_ID p" src/basenames.txt)

#Create output directory for the results.
# NOTE: the -p option will not only create parental directories if needed, but also will not
# throw an error or overwrite the directory if it already exists
mkdir -p results/featureCounts

#run featureCounts
echo "start featureCounts"

featureCounts -T 1 -s 2 -g gene_id -t exon \
-o results/featureCounts/${line}_featCounts.txt \
-a data/genome/mouse_chr12.gtf \
results/star/${line}_Aligned.sortedByCoord.out.bam

# NOTE: The -t option describes the "feature" that this software will look for in our GTF file.
# In our case it will identify all the exons listed in our GTF and only count reads that align
# to an exon at least partially. The -g option describes the "meta-feature" that should
# also be present in our GTF. Once, featureCounts counts all the reads aligning to exons, it 
# will group those counts by "gene_id" (i.e. by gene). "gene_id" is often chosen as a meta-feature
# for the -g option because it usually describes an ID that is unique for every gene in GTF files..  

echo "end featureCounts"
