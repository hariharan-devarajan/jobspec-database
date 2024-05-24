#!/bin/bash

#SBATCH --job-name=03_SNPj_filterdp_CERW_reducedref
#SBATCH --partition=comp01
#SBATCH --output=03_SNPj_filterdp_CERW_reducedref_%j.txt
#SBATCH --error=03_SNPj_filterdp_CERW_reducedref_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alix.matthews@smail.astate.edu  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00

##----------------------------------------------------------------------------------------
## LOAD MODULES

# I created a bcftools env with the newest version (1.15.1) using the following:

# ssh pinnacle-l6
# module load python/anaconda-3.9
# source /share/apps/bin/conda-3.9.sh
# mamba create -n BCFTools bcftools

# so load the stuff needed to activate that environment below 

module load python/anaconda-3.9
source /share/apps/bin/conda-3.9.sh
conda activate BCFTools


cd /local_scratch/$SLURM_JOB_ID/


##----------------------------------------------------------------------------------------
## "ADJUSTABLES"

## SPECIES OF INTEREST
SPP=CERW 

## MAIN DIR
PROJECT_DIR=/scrfs/storage/amatthews/20210816_projects/20210816_snp

## REF DIR, REF, AND REFTYPE
# it's a version of this reference, best assembly from spades on 20211014 of PROW 981 R1: /scrfs/storage/amatthews/20210816_projects/20210816_exp/01_ASSEMBLY/spades_20211014/PROW_981_R1_TGCTGTGA-GAATCGTG/scaffolds.fasta.... but I have removed the microbe seqs and all contigs <5000bp (scaffolds_reduced_contigs_kept.fasta)

REF_DIR=$PROJECT_DIR/02_IndexRef/ref_full
REF=scaffolds_reduced_contigs_kept
REFTYPE=ref_reduced


## SNP DIR 
SNP_DIR=$PROJECT_DIR/03_SNP_20220513

## RESULTS DIR PER SPECIES
RESULTS_DIR=$SNP_DIR/RESULTS_${REFTYPE}_$SPP



##----------------------------------------------------------------------------------------
## Filter by seq depth, several different thresholds. I already put the max-depth as 50 in the 'SNPe_mpileup' slurm. If I do two expressions in the +setGT plugin, it does not work as expected... so need to trust that the max-depth setting as 50 was implemented correctly in the mpileup command. I believe it was.

## So apparently there is a bug for the +setGT command: https://github.com/samtools/bcftools/issues/1607. It says it was fixed in the newest version (the one I'm using), but maybe I am just thinking about the --include vs. --exclude backwards.

## The solution is use the filtering expression which kind of the opposite of what you would think. So here I am filtering by DP. I want to change the GT for a sample to ./. if DP is less than 5/10/15/20. Intuitively, I would put {--exclude 'FMT/DP<5}; however, that is not the case (by checking the resulting .vcf files, which did not do what I wanted at all - it was changing everything backwards or in some weird pattern). Instead, here, we are using {--include 'FORMAT/DP<5}. The resulting .vcf files are behaving as expected.


bcftools +setGT ${RESULTS_DIR}/vcf/${SPP}_ALL_renamed_q30_minac1_maf05.vcf.gz --output-type z --output ${RESULTS_DIR}/vcf/${SPP}_ALL_renamed_q30_minac1_maf05_dp05.vcf.gz -- --target-gt q --new-gt . --include 'FORMAT/DP<5' 

bcftools +setGT ${RESULTS_DIR}/vcf/${SPP}_ALL_renamed_q30_minac1_maf05.vcf.gz --output-type z --output ${RESULTS_DIR}/vcf/${SPP}_ALL_renamed_q30_minac1_maf05_dp10.vcf.gz -- --target-gt q --new-gt . --include 'FORMAT/DP<10' 

bcftools +setGT ${RESULTS_DIR}/vcf/${SPP}_ALL_renamed_q30_minac1_maf05.vcf.gz --output-type z --output ${RESULTS_DIR}/vcf/${SPP}_ALL_renamed_q30_minac1_maf05_dp15.vcf.gz -- --target-gt q --new-gt . --include 'FORMAT/DP<15' 

bcftools +setGT ${RESULTS_DIR}/vcf/${SPP}_ALL_renamed_q30_minac1_maf05.vcf.gz --output-type z --output ${RESULTS_DIR}/vcf/${SPP}_ALL_renamed_q30_minac1_maf05_dp20.vcf.gz -- --target-gt q --new-gt . --include 'FORMAT/DP<20' 






