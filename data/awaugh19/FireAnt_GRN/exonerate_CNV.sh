#!/bin/bash
#SBATCH --job-name=exonerate_CNV
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24gb
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/ahw22099/FireAnt_GRN/scripts/exonerate_CNV.log.%j
#SBATCH --error=/scratch/ahw22099/FireAnt_GRN/scripts/exonerate_CNV.err.%j
#SBATCH --mail-user=ahw22099@uga.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-261

# Load necessary modules
module load Exonerate/2.4.0-GCC-12.2.0

# Change to working directory
cd /scratch/ahw22099/FireAnt_GRN/Fontana2020_CNV

CNV_genes="/scratch/ahw22099/FireAnt_GRN/Fontana2020_CNV/CNV_genes"
if [ ! -d $CNV_genes ]; then
  mkdir -p $CNV_genes
fi

# Define input files
#awk -v dir="$CNV_genes" '/^>/{s=++d".fa"} {print > s}' Sb-vs-SB_CNV_genes_Sinv.fasta
#mv *.fa ./CNV_genes

#make sample list for array job
fasta_num_list=($(<CNV_exo_input_list.txt))
fasta_num=${fasta_num_list[${SLURM_ARRAY_TASK_ID}]}

cd $CNV_genes
num=$(basename "$fasta_num" .fa)
OUTPUT_FILE="${num}_exonerate_output.txt"

exonerate --model est2genome --showtargetgff --showalignment --showvulgar --bestn 5 --query $fasta_num --target /scratch/ahw22099/FireAnt_GRN/Fontana2020_CNV/GCF_016802725.1_UNIL_Sinv_3.0_genomic.fasta --verbose 1 > "$OUTPUT_FILE"
echo "Exonerate alignment for $gene_name completed. Results are saved in $OUTPUT_FILE"
done
