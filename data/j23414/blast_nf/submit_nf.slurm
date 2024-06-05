#! /usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --job-name=time_blast
#SBATCH --output=R-%x.%J.out
#SBATCH --error=R-%x.%J.err
# --mail-user=username@email.com
# --mail-type=begin
# --mail-type=end
# --account=project_name

module load nextflow
module load blast       # or ncbi

nextflow run main.nf \
  --db "/path/to/nt.fasta" \
  --query "*.fasta" \
  --options '-outfmt 6 -num_alignments 1'
  --threads 50