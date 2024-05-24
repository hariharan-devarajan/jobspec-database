#!/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=3:00:00
#SBATCH --partition=shortq

VCF_FILES=$( find "/mnt/beegfs/scratch/m_michel/DATA/" -type f -name "*.vcf.gz" )

> rmInfo.log

for VCF in ${VCF_FILES}; do
  NEW=$( cut -d/ -f7 <<<"$VCF" )
  OUTPUT=/mnt/beegfs/scratch/m_michel/VCFRM/${NEW}.vcf
  tabix ${VCF} -f
  bcftools filter -i 'QUAL>20 && FORMAT/DP>10 && (FORMAT/AD[0:1]*100)/(FORMAT/AD[0:0]+FORMAT/AD[0:1]) >= 5  && FORMAT/AD[0:1]>3 ' ${VCF} > ${OUTPUT}
  SnpSift  rmInfo ${OUTPUT} AC VT DP FREQ_HTZ FREQ_HOM DB H3 H3_AF 1000G 1000G_AF EVS EVS_AF Dels IMPRECISE ANN GeneNameRefSeq EVS_MAF EVS_CA mir_ACC clinvar clinvar_CLNACC IG IG_hom-het ID ref_upstream ref/indel ref_downstream max_gtype Qmax_gtype depth alt_reads indel_reads other_reads repeat_unit ref_repeat_count indel_repeat_count bcalls_used bcalls_filt max_gt Qmax_gt A_used C_used G_used T_used Statut ANN Kaviar clinvar clinvar_CLNSICC IG IG_AF Kaviar_AF cosmic cosmic_FATHMM PhyloP PhastCons DisGeNet GnomAD_Genome GnomAD_Genome_AF GnomAD_Genome_AF_AFR GnomAD_Genome_AF_AMR GnomAD_Genome_AF_ASJ GnomAD_Genome_AF_EAS GnomAD_Genome_AF_FIN GnomAD_Genome_AF_NFE GnomAD_Genome_AF_OTH GnomAD_Genome_AF_SAS GnomAD_Genome_AF_POPMAX FATHMM_NCS FATHMM FATHMM_NCG FATHMM_CS FATHMM_CG DANN DANN_Score BaseQRankSum ClippingRankSum DS END ExcessHet FS HaplotypeScore InbreedingCoeff MLEAC MLEAF MQ MQRankSum NEGATIVE_TRAIN_SITE POSITIVE_TRAIN_SITE QD RAW_MQ ReadPosRankSum SOR VQSLOD culprit 2>> rmInfo.log | egrep -v "^##INFO" | bgzip  > ${OUTPUT}.gz
  tabix ${OUTPUT}.gz -f
 done;
