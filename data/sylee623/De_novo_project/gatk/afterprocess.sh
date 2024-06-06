#PBS -lvnode=biglab3
#PBS -N Merge
#PBS -j oe
#PBS -o Merge.log

/share/apps/programs/bcftools/1.7/bin/bcftools concat -o /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/total.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr1.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr2.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr3.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr4.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr5.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr6.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr7.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr8.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr9.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr10.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr11.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr12.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr13.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr14.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr15.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr16.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr17.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr18.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr19.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr20.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr21.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chr22.hc.g.vcf.gz /lustre/export/home/sylee/Hanchinese/try3/result/gatk/HaplotypeCaller/chrX.hc.g.vcf.gz

work_dir=/lustre/export/home/sylee/Hanchinese/try3

#reference
reference=/lustre/export/home/sylee/ucsc_hg19/ucsc.hg19.fasta
GATK_bundle=/lustre/export/home/sylee/ucsc_hg19

TILEDB_DISABLE_FILE_LOCKING=1

cd ${work_dir}/result/gatk

/lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx8g -Xms8g" IndexFeatureFile -I /lustre/export/home/sylee/Hanchinese/try3/result/gatk/total.hc.g.vcf.gz

ls *.GenotypeGVCF.g.vcf| while read id; do
/lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx4g -Xms4g" GenomicsDBImport \
    -V /lustre/export/home/sylee/Hanchinese/try1/material/renamed_whole_cFather.gvcf \
    -V /lustre/export/home/sylee/Hanchinese/try1/material/cm_renamed.gvcf \
    -V /lustre/export/home/sylee/Hanchinese/try3/result/gatk/total.hc.g.vcf.gz \
    --genomicsdb-workspace-path /lustre/export/home/sylee/Hanchinese/try3/result/gatk/database \
    --tmp-dir /home/sylee/tmp/ \
    --reader-threads 5 \
done;

ls *.GenotypeGVCF.g.vcf| while read id; do
/lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx5g -Xms5g" \
    GenotypeGVCFs \
        -R ${reference} \
        -V /lustre/export/home/sylee/Hanchinese/try3/result/gatk/database \
        -O /lustre/export/home/sylee/Hanchinese/try3/result/gatk/total.GenotypeGVCF.g.vcf \
        --tmp-dir=/lustre/export/home/sylee/tmp \
done

ls *.GenotypeGVCF.g.vcf| while read id; do
/lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx8g -Xms8g" \
    VariantRecalibrator \
    -V ${work_dir}/result/gatk/$id \
    --trust-all-polymorphic \
    -tranche 70.0 \
    -an FS -an ReadPosRankSum -an MQRankSum -an QD -an SOR -an DP \
    -mode INDEL \
    --max-gaussians 4 \
    -resource:mills,known=false,training=true,truth=true,prior=12 /lustre/export/home/sylee/ucsc_hg19/Mills_and_1000G_gold_standard.indels.hg19.sites.vcf \
    -resource:axiomPoly,known=false,training=true,truth=false,prior=10 /lustre/export/home/sylee/ucsc_hg19/Axiom_Exome_Plus.genotypes.all_populations.poly.vcf.gz \
    -resource:dbsnp,known=true,training=false,truth=false,prior=2 /lustre/export/home/sylee/ucsc_hg19/dbsnp_138.hg19.vcf \
    -O ${work_dir}/result/gatk/VariantRecalibrator/indels.recal \
    --tranches-file ${work_dir}/result/gatk/VariantRecalibrator/indels.tranches
done;

ls *.GenotypeGVCF.g.vcf| while read id; do
/lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx8g -Xms8g" VariantRecalibrator \
    -V ${work_dir}/result/gatk/$id \
    --trust-all-polymorphic \
    -tranche 70.0 \
    -an QD -an MQRankSum -an ReadPosRankSum -an FS -an MQ -an SOR -an DP \
    -mode SNP \
    --max-gaussians 6 \
    -resource:hapmap,known=false,training=true,truth=true,prior=15 /lustre/export/home/sylee/ucsc_hg19/hapmap_3.3.hg19.sites.vcf \
    -resource:omni,known=false,training=true,truth=true,prior=12 /lustre/export/home/sylee/ucsc_hg19/1000G_omni2.5.hg19.sites.vcf \
    -resource:1000G,known=false,training=true,truth=false,prior=10 /lustre/export/home/sylee/ucsc_hg19/1000G_phase1.indels.hg19.sites.vcf \
    -resource:dbsnp,known=true,training=false,truth=false,prior=7 /lustre/export/home/sylee/ucsc_hg19/dbsnp_138.hg19.vcf \
    -O ${work_dir}/result/gatk/result/gatk/VariantRecalibrator/snps.recal \
    --tranches-file ${work_dir}/result/gatk/result/gatk/VariantRecalibrator/snps.tranches
done;

ls *.GenotypeGVCF.g.vcf| while read id; do
/lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx5g -Xms5g" \
    ApplyVQSR \
    -V ${work_dir}/result/gatk/$id \
    --recal-file ${work_dir}/result/gatk/VariantRecalibrator/indels.recal \
    --tranches-file ${work_dir}/result/gatk/VariantRecalibrator/indels.tranches \
    --truth-sensitivity-filter-level 70 \
    --create-output-variant-index true \
    -mode INDEL \
    -O ${work_dir}/result/gatk/indels_ApplyVQSR.vcf
done;

ls *.GenotypeGVCF.g.vcf| while read id; do
/lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx8g -Xms8g" \
    ApplyVQSR \
    -V ${work_dir}/result/gatk/indels_ApplyVQSR.vcf \
    --recal-file ${work_dir}/result/gatk/result/gatk/VariantRecalibrator/snps.recal\
    --tranches-file ${work_dir}/result/gatk/result/gatk/VariantRecalibrator/snps.tranches \
    --truth-sensitivity-filter-level 70 \
    --create-output-variant-index true \
    -mode SNP \
    -O ${work_dir}/result/gatk/result/gatk/VariantRecalibrator/snp_applyVQSR.vcf
done;

ls *.snp_applyVQSR.vcf| while read id; do
    /lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx8g -Xms8g" \
        CalculateGenotypePosteriors \
        -R ${reference} \
        -V ${work_dir}/result/gatk/$id \
        -O ${work_dir}/result/gatk/result/gatk/calculategenotypeposteriors.vcf \
        -ped /lustre/export/home/sylee/Hanchinese/filter/wholechromosome/family.ped
done;

ls *.calculategenotypeposteriors.vcf| while read id; do
    /lustre/export/home/sylee/programfile/gatk-4.1.7.0/gatk --java-options "-Xmx8g -Xms8g" \
        VariantFiltration \
        -R ${reference} \
        -V ${work_dir}/result/gatk/$id \
        --genotype-filter-expression "GQ < 20" --genotype-filter-name "lowGQ" \
        -O ${work_dir}/result/gatk/result/gatk/gq20filtered.vcf
done;

ls *.gq20filtered.vcf| while read id; do
    /lustre/export/home/sylee/gatk-4.1.7.0/gatk --java-options "-Xmx8g -Xms8g" \
        gatk SelectVariants \
        -V input.vcf \
        --mendelian-violation true \
        --mendelian-violation-qual-threshold 20 \
        -ped family.ped \
        --exclude-filtered true \
        -O output.vcf
done;


ls *.gq20filtered.vcf| while read id; do
gatk SelectVariants \
-V input.vcf \
--restrict-alleles-to BIALLELIC \
-O output.biallelic.vcf
done;


ls *.gq20filtered.vcf| while read id; do
gatk SelectVariants \
     -R ucsc.hg19.fasta \
     -V VQSR_finished.vcf \
     --select-type-to-include SNP \
     -select 'vc.getGenotype("Chinese").isHet()' \
     -O SNVonly.vcf \
done;


ls *.gq20filtered.vcf| while read id; do
gatk SelectVariants \
     -R ucsc.hg19.fasta \
     -V VQSR_finished.vcf \
     --select-type-to-include INDEL \
     -O Indelonly.vcf \
done;


ls *.gq20filtered.vcf| while read id; do
gatk VariantAnnotator \
-R data/ucsc.hg19.fasta \
-V input.vcf \
-A AlleleFraction \
-O annotated.vcf
done;