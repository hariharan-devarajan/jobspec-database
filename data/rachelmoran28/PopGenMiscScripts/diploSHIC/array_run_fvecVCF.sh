#!/bin/bash
#SBATCH -p ram256g,ram1t,amd2tb,amdlarge,amdsmall,small,astyanax,cavefish
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=230gb
#SBATCH -t 28:00:00
#SBATCH -J fvecVCF
#SBATCH -e CMS_fvecVCF.err
#SBATCH -o CMS_fvecVCF.out
#SBATCH --array=1001-2360

#2360 total

source /home/mcgaughs/rmoran/miniconda3/etc/profile.d/conda.sh
conda activate diplo

#generating feature vector stats
cd /home/mcgaughs/shared/Software/diploSHIC

#vcf="/home/mcgaughs/shared/Datasets/all_sites_LARGE_vcfs/filtered_surfacefish/combined_filtered/246Indvs_RepsMasked/phased_masked/Split_All_Scafolds_VCFs"

#vcf="/home/mcgaughs/shared/Datasets/all_sites_LARGE_vcfs/filtered_surfacefish/combined_filtered/246Indvs_RepsMasked/phased_masked/biallelicSNPs_wInvar_noIndels_phased_masked_ExcludeHet"


#HardFiltered SNPs
#vcf="/home/mcgaughs/shared/Datasets/all_sites_LARGE_vcfs/filtered_surfacefish/filtered_snps/HardFilteredSNPs_250Samples"
vcf="/home/mcgaughs/shared/Datasets/all_sites_LARGE_vcfs/filtered_surfacefish/combined_filtered/250_samples"

Pop="CabMoroSurface"

#CMD_LIST="HardFiltered_All_wInvar_Commands_fvecVCF_5kb_803scaffs_w_data.txt"
#CMD_LIST="HardFiltered_All_wInvar_Commands_fvecVCF_5kb_Escon1797.txt"
CMD_LIST="HardFiltered_All_wInvar_Commands_fvecVCF_5kb.txt"
#CMD_LIST="HardFiltered_All_Commands_fvecVCF.txt"

CMD="$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${CMD_LIST})"
eval ${CMD}

