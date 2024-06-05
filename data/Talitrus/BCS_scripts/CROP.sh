#!/bin/bash
#SBATCH -N 1
#SBATCH -t 2-00:00:00
#SBATCH -J CROP
#SBATCH -p defq,short
#SBATCH --mail-type=all
#SBATCH --mail-user=bnguyen@gwu.edu
#SBATCH -o out_err_files/CROP_%A_%a.out
#SBATCH -e out_err_files/CROP_%A_%a.err
name1=$(sed -n "$SLURM_ARRAY_TASK_ID"p seq_list.txt)
cd ../data/seq
#module load mothur
module load gsl/gcc/2.3
module load CROP
#mothur "#count.seqs(name=CombinedRuns_denoised_contiged.trim.pick.names, group=CombinedRuns_denoised_contiged.pick.groups); pre.cluster(fasta=macse.fasta,count=CombinedRuns_denoised_contiged.trim.pick.count_table,diffs=3)"
#mothur "#deunique.seqs(fasta=macse.precluster.pick.pick.fasta, count=macse.precluster.pick.pick.count_table)"
CROP -i macse.precluster.pick.pick.redundant.fasta -o macse.precluster.pick.pick.redundant_CROP -l 3 -u 4 -z 450 -b 850
#adjust b according to the manual, z according to amplicon length
#consider using new OptiClust clustering built into mothur
