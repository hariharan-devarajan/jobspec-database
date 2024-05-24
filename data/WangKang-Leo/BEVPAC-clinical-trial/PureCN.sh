#################Re-run Mutect2#######################
cd /proj/sens2019581/nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/Mutect2_germline_added
module load bioinfo-tools 
module load Nextflow/21.04.1 
module load nf-core/1.14 
module load iGenomes/latest
export NXF_OFFLINE='TRUE'
export NXF_HOME="/castor/project/proj/nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/"
export PATH=${NXF_HOME}:${PATH}
export NXF_TEMP=$SNIC_TMP
export NXF_LAUNCHER=$SNIC_TMP
export NXF_SINGULARITY_CACHEDIR="/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/"
nextflow run /castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/workflow/main.withgermline.nf \
-profile uppmax \
-with-singularity "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/singularity-images/nf-core-sarek-2.7.1.simg" \
--custom_config_base "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/configs" \
-c "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-data/sarek.custom.config" \
--project sens2019581 \
--input "/proj/sens2019581/nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/results/Preprocessing/TSV/recalibrated.tsv" \
--genome GRCh38 \
--germline_resource "$IGENOMES_DATA/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz" \
--germline_resource_index "$IGENOMES_DATA/Homo_sapiens/GATK/GRCh38/Annotation/GermlineResource/gnomAD.r2.1.1.GRCh38.PASS.AC.AF.only.vcf.gz.tbi" \
--pon "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz" \
--pon_index "/castor/project/proj_nobackup/references/Homo_sapiens/GATK/GRCh38/Annotation/GATKBundle/1000g_pon.hg38.vcf.gz.tbi" \
--step 'variant_calling' \
--tools 'mutect2' \
--target_bed "/castor/project/proj_nobackup/nf-core2/nf-core-sarek-2.7.1/BEVPAC-data/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed" \
-resume

ln -s /proj/sens2019581/nobackup/nf-core2/nf-core-sarek-2.7.1/Sarek-results/Sarek-BEVPAC_WES/Mutect2_germline_added/results/VariantCalling/*vs*/Mutect2/Mutect2_filtered*vcf.gz /proj/sens2019581/nobackup/PureCN/data

cd /proj/sens2019581/nobackup/PureCN/data

for file in *.vcf.gz;
do 
ID=${file#Mutect2_filtered_}
ID=${ID%%vs*}
ID=$(basename $ID _).vcf
echo unzip $ID
gunzip -c "$file">"$ID"
done

###########################################
###############PureCN scratch##############
###########################################
# Step1 reference files

#!/bin/bash -l
#SBATCH -A sens2019581
#SBATCH -p node
#SBATCH -n 2
#SBATCH -t 30:00:00
#SBATCH -J PureCN
module load R/4.2.1
module load R_packages/4.2.1
export PURECN="/sw/apps/R_packages/4.2.1/bianca/PureCN/extdata"
export OUT_REF="/proj/sens2019581/nobackup/PureCN/reference_files"
Rscript $PURECN/IntervalFile.R --in-file /proj/sens2019581/nobackup/PureCN/reference_files/Twist_Exome_RefSeq_targets_hg38_100bp_padding.bed \
    --fasta /sw/data/uppnex/ToolBox/hg38bundle/Homo_sapiens_assembly38.fasta \
    --genome hg38 \
    --mappability /proj/sens2019581/nobackup/PureCN/reference_files/GCA_000001405.15_GRCh38_no_alt_analysis_set_100.bw \
    --out-file $OUT_REF/Twist_Exome_RefSeq_targets_hg38_intervals.txt

# Step2 Coverage

#!/bin/bash -l
#SBATCH -A sens2019581
#SBATCH -p node
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH -J coverage_normal
module load bioinfo-tools
module load R/4.2.1
module load R_packages/4.2.1
cd /proj/sens2019581/nobackup/PureCN/data/Normal
export PURECN="/sw/apps/R_packages/4.2.1/bianca/PureCN/extdata"
export OUT_REF="/proj/sens2019581/nobackup/PureCN/reference_files"
export OUT="/proj/sens2019581/nobackup/PureCN/data/Normal"
Rscript $PURECN/Coverage.R --out-dir $OUT \
    --bam /proj/sens2019581/nobackup/PureCN/data/normal_bam.list \
    --intervals $OUT_REF/Twist_Exome_RefSeq_targets_hg38_intervals.txt \
    --cores 64 --parallel


#!/bin/bash -l
#SBATCH -A sens2019581
#SBATCH -p node
#SBATCH -n 4
#SBATCH -t 12:00:00
#SBATCH -J coverage_tumor
module load bioinfo-tools
module load R/4.2.1
module load R_packages/4.2.1
cd /proj/sens2019581/nobackup/PureCN/data/Tumor
export PURECN="/sw/apps/R_packages/4.2.1/bianca/PureCN/extdata"
export OUT_REF="/proj/sens2019581/nobackup/PureCN/reference_files"
export OUT="/proj/sens2019581/nobackup/PureCN/data/Tumor"
Rscript $PURECN/Coverage.R --out-dir $OUT \
    --bam /proj/sens2019581/nobackup/PureCN/data/tumor_bam.list \
    --intervals $OUT_REF/Twist_Exome_RefSeq_targets_hg38_intervals.txt \
    --cores 64 --parallel

# Rename Coverage output
for file in *txt
do 
echo $file
sample=$(basename $file .recal.gz.txt)
mv $file $sample".gz.txt"
done

# NormalDB
cd /proj/sens2019581/nobackup/PureCN/data/Normal
ls -a *_loess.txt.gz | cat > BEVPAC_normal_coverages.list

#!/bin/bash -l
#SBATCH -A sens2019581
#SBATCH -p node
#SBATCH -n 6
#SBATCH -t 12:00:00
#SBATCH -J NormalDB
module load bioinfo-tools
module load R/4.2.1
module load R_packages/4.2.1
cd /proj/sens2019581/nobackup/PureCN/data/Normal
export PURECN="/sw/apps/R_packages/4.2.1/bianca/PureCN/extdata"
export OUT_REF="/proj/sens2019581/nobackup/PureCN/reference_files"
Rscript $PURECN/NormalDB.R --out-dir $OUT_REF \
    --coverage-files /proj/sens2019581/nobackup/PureCN/data/Normal/BEVPAC_normal_coverages.list \
    --genome hg38 --assay Twist_Exome


#PureCN run
#!/bin/bash -l
#SBATCH -A sens2019581
#SBATCH -p node
#SBATCH -n 2
#SBATCH -t 48:00:00
#SBATCH -J PureCN
module load bioinfo-tools
module load R/4.2.1
module load R_packages/4.2.1
export PURECN="/sw/apps/R_packages/4.2.1/bianca/PureCN/extdata"
export OUT_REF="/proj/sens2019581/nobackup/PureCN/reference_files"
export OUT="/proj/sens2019581/nobackup/PureCN/results"
for ((i=1; i<=18; i++)); #header(1)  # i=2; i<=194
do
Sample=$(awk -v i=$i 'NR==i{print $2}' /proj/sens2019581/nobackup/PureCN/data/BEVPAC_WESpair_baseline.txt)
mkdir $OUT/$Sample
Rscript $PURECN/PureCN.R --out $OUT/$Sample \
    --tumor /proj/sens2019581/nobackup/PureCN/data/Tumor/$Sample".recal_coverage_loess.txt.gz" \
    --sampleid $Sample \
    --vcf /proj/sens2019581/nobackup/PureCN/data/$Sample".vcf" \
    --fun-segmentation PSCBS \
    --normaldb $OUT_REF/normalDB_Twist_Exome_hg38.rds \
    --intervals $OUT_REF/Twist_Exome_RefSeq_targets_hg38_intervals.txt \
    --snp-blacklist $OUT_REF/hg38-blacklist.v2.bed \
    --genome hg38 \
    --minpurity 0.05 --max-copy-number 8 \
    --rds=rda \
    --max-non-clonal 0.3 \
    --model betabin \
    --force --post-optimize --seed 123 \
    --cores 32
done


cd /proj/sens2019581/nobackup/PureCN/results
cat */*loh.csv>BEVPAC_PureCN_LOH.csv
cat */*variants.csv>BEVPAC_PureCN_variants.csv
cat */*dnacopy.seg>BEVPAC_PureCN_seg.csv
cat */!(*loh*|*variants*|*genes*|*amplification_pvalues*).csv>BEVPAC_PureCN_purity.csv
cp BEVPAC_PureCN* /proj/nobackup/sens2019581/wharf/kangwang/kangwang-sens2019581/PureCN/



