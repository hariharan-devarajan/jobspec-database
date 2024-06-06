#have to index the files first
/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk IndexFeatureFile -F Holly_11B_S15_R1_001_trimmed_tophat_out.calls.vcf
/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk IndexFeatureFile -F Sample2.sorted.fixed.marked.realigned.chgchrm16.vcf
/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk IndexFeatureFile -F Holly_2B_S7_R1_001_trimmed_tophat_out.calls.vcf

#build dict for fasta ref file
/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk CreateSequenceDictionary.jar R= genome.fa O= genome.dict

/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk SelectVariants -V Holly-GC-11-redo_S6_R1_001_trimmed_tophat_out.calls.vcf -conc Holly_11B_S15_R1_001_trimmed_tophat_out.calls.vcf -O 11_A_B_conc.vcf


/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk SelectVariants  -R genome.fa --exclude-non-variants -V Sample2.sorted.fixed.marked.realigned.chgchrm16.vcf -conc Holly_2A_S1_R1_001_trimmed_tophat_out.calls.vcf -O 2_test_conc.vcf

/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk SelectVariants -R genome.fa -V Holly_2A_S1_R1_001_trimmed_tophat_out.calls.vcf -disc Sample2.sorted.fixed.marked.realigned.chgchrm16.vcf -O 2_test_disc.vcf

/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk SelectVariants -R genome.fa -V Holly-GC-11-redo_S6_R1_001_trimmed_tophat_out.calls.vcf -disc Sample2.sorted.fixed.marked.realigned.chgchrm16.vcf  -O 11_2_disc.vcf

#change chromosome names in Megan's VCFs
#awk -F'\t' '{
#if ($1=="ref|NC_001133|") print $0,"\t","I";
#if ($1=="ref|NC_001134|") print $0,"\t","II";
#if ($1=="ref|NC_001135|") print $0,"\t","III";
#if ($1=="ref|NC_001136|") print $0,"\t","IV";
#if ($1=="ref|NC_001137|") print $0,"\t","V";
#if ($1=="ref|NC_001138|") print $0,"\t","VI";
#if ($1=="ref|NC_001139|") print $0,"\t","VII";
#if ($1=="ref|NC_001140|") print $0,"\t","VIII";
#if ($1=="ref|NC_001141|") print $0,"\t","IX";
#if ($1=="ref|NC_001142|") print $0,"\t","X";
#if ($1=="ref|NC_001143|") print $0,"\t","XI";
#if ($1=="ref|NC_001144|") print $0,"\t","XII";
#if ($1=="ref|NC_001145|") print $0,"\t","XIII";
#if ($1=="ref|NC_001146|") print $0,"\t","XIV";
#if ($1=="ref|NC_001147|") print $0,"\t","XV";
#if ($1=="ref|NC_001148|") print $0,"\t","XVI";

#}' Sample2.sorted.fixed.marked.realigned.vcf > Sample2.sorted.fixed.marked.realigned.chgchrm.vcf

#sed -i -e 's/ref|NC_001133|/I/g' Sample2.sorted.fixed.marked.realigned.vcf > Sample2.sorted.fixed.marked.realigned.chgchrm.vcf


#for file in ./*.vcf
#do
#FBASE=$(basename $file .vcf)
#BASE=${FBASE%.vcf}
#sed 's/ref|NC_001133|/I/g' <${BASE}.vcf >${BASE}.chgchrm.vcf
#sed 's/ref|NC_001134|/II/g'<${BASE}.chgchrm.vcf >${BASE}.chgchrm2.vcf
#sed 's/ref|NC_001135|/III/g' <${BASE}.chgchrm2.vcf >${BASE}.chgchrm3.vcf
#sed 's/ref|NC_001136|/IV/g' <${BASE}.chgchrm3.vcf >${BASE}.chgchrm4.vcf
#sed 's/ref|NC_001137|/V/g' <${BASE}.chgchrm4.vcf >${BASE}.chgchrm5.vcf
#sed 's/ref|NC_001138|/VI/g' <${BASE}.chgchrm5.vcf >${BASE}.chgchrm6.vcf
#sed 's/ref|NC_001139|/VII/g' <${BASE}.chgchrm6.vcf >${BASE}.chgchrm7.vcf
#sed 's/ref|NC_001140|/VIII/g' <${BASE}.chgchrm7.vcf >${BASE}.chgchrm8.vcf
#sed 's/ref|NC_001141|/IX/g' <${BASE}.chgchrm8.vcf >${BASE}.chgchrm9.vcf
#sed 's/ref|NC_001142|/X/g' <${BASE}.chgchrm9.vcf >${BASE}.chgchrm10.vcf
#sed 's/ref|NC_001143|/XI/g' <${BASE}.chgchrm10.vcf >${BASE}.chgchrm11.vcf
#sed 's/ref|NC_001144|/XII/g' <${BASE}.chgchrm11.vcf >${BASE}.chgchrm12.vcf
#sed 's/ref|NC_001145|/XIII/g' <${BASE}.chgchrm12.vcf >${BASE}.chgchrm13.vcf
#sed 's/ref|NC_001146|/XIV/g' <${BASE}.chgchrm13.vcf >${BASE}.chgchrm14.vcf
#sed 's/ref|NC_001147|/XV/g' <${BASE}.chgchrm14.vcf >${BASE}.chgchrm15.vcf
#sed 's/ref|NC_001148|/XVI/g' <${BASE}.chgchrm15.vcf >${BASE}.chgchrm16.vcf
#done


for file in ./*.txt
do
FBASE=$(basename $file .txt)
BASE=${FBASE%.txt}
sed 's/ref|NC_001133|/I/g' <${BASE}.txt >${BASE}.chgchrm.txt
sed 's/ref|NC_001134|/II/g'<${BASE}.chgchrm.txt >${BASE}.chgchrm2.txt
sed 's/ref|NC_001135|/III/g' <${BASE}.chgchrm2.txt >${BASE}.chgchrm3.txt
sed 's/ref|NC_001136|/IV/g' <${BASE}.chgchrm3.txt >${BASE}.chgchrm4.txt
sed 's/ref|NC_001137|/V/g' <${BASE}.chgchrm4.txt >${BASE}.chgchrm5.txt
sed 's/ref|NC_001138|/VI/g' <${BASE}.chgchrm5.txt >${BASE}.chgchrm6.txt
sed 's/ref|NC_001139|/VII/g' <${BASE}.chgchrm6.txt >${BASE}.chgchrm7.txt
sed 's/ref|NC_001140|/VIII/g' <${BASE}.chgchrm7.txt >${BASE}.chgchrm8.txt
sed 's/ref|NC_001141|/IX/g' <${BASE}.chgchrm8.txt >${BASE}.chgchrm9.txt
sed 's/ref|NC_001142|/X/g' <${BASE}.chgchrm9.txt >${BASE}.chgchrm10.txt
sed 's/ref|NC_001143|/XI/g' <${BASE}.chgchrm10.txt >${BASE}.chgchrm11.txt
sed 's/ref|NC_001144|/XII/g' <${BASE}.chgchrm11.txt >${BASE}.chgchrm12.txt
sed 's/ref|NC_001145|/XIII/g' <${BASE}.chgchrm12.txt >${BASE}.chgchrm13.txt
sed 's/ref|NC_001146|/XIV/g' <${BASE}.chgchrm13.txt >${BASE}.chgchrm14.txt
sed 's/ref|NC_001147|/XV/g' <${BASE}.chgchrm14.txt >${BASE}.chgchrm15.txt
sed 's/ref|NC_001148|/XVI/g' <${BASE}.chgchrm15.txt >${BASE}.chgchrm16.txt
done


#awk 'NR==FNR {a[$1]; next} $2 in a {print}'  Holly_2B_S7_R1_001_trimmed_tophat_out.calls.vcf GC_LOH_datafileB.chgchrm16.txt > test.txt



#####################################
#works to find the lines in file 1 that also match the columns given in file 2, outputs those lines
#matching the vcf file to the LOH file, to see which samples my sample matches to
#need to get rid of duplicates next, since there are some that match to all of them
for i in ./*.vcf; do
    awk 'FNR==NR{arr[$1,$2];next}(($2,$5) in arr)' "$i" GC_LOH_datafileB.chgchrm16.txt  >"$i.matched.txt"
done

awk '!seen[$5]++' Holly-GC-8-redo_S5_R1_001_trimmed_tophat_out.calls.vcf.matched.txt > test.short.txt

sort -u -t'\t' -k2,5 Holly-GC-8-redo_S5_R1_001_trimmed_tophat_out.calls.vcf.matched.txt > test.short.txt

awk 'FNR==NR{arr[$1,$2];next}(($2,$5) in arr)' Holly_2B_S7_R1_001_trimmed_tophat_out.calls.vcf GC_LOH_datafileB.chgchrm16.txt  > test.txt

module load bcftools/1.6

bcftools view Holly_9B_S14_R1_001_trimmed_tophat_out.calls.vcf  -Oz -o Holly_9B_S14_R1_001_trimmed_tophat_out.calls.vcf.gz
 bcftools index Holly_9B_S14_R1_001_trimmed_tophat_out.calls.vcf.gz


bcftools isec -p line9.test -n=2 -w1 Holly_9A_S5_R1_001_trimmed_tophat_out.calls.vcf.gz Holly_9B_S14_R1_001_trimmed_tophat_out.calls.vcf.gz



#!/bin/bash

#PBS -N j_Vcftools
#PBS -q batch
#PBS -l nodes=1:ppn=1:AMD
#PBS -l walltime=480:00:00
#PBS -l mem=50gb

module load vcftools/0.1.12b
cd $PBS_O_WORKDIR
time /usr/local/apps/vcftools/0.1.12b/bin/vcftools




#sed 's/ref|NC_001133|/I/g' <${BASE}.vcf >${BASE}.chgchrm.vcf
#sed 's/ref|NC_001134|/II/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.2.vcf
#sed 's/ref|NC_001135|/III/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001136|/IV/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001137|/V/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001138|/VI/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001139|/VII/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001140|/VIII/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001141|/IX/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001142|/X/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001143|/XI/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001144|/XII/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001145|/XIII/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001146|/XIV/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001147|/XV/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf
#sed 's/ref|NC_001148|/XVI/g' <Sample2.sorted.fixed.marked.realigned.chgchrm.vcf >Sample2.sorted.fixed.marked.realigned.chgchrm.vcf






Holly_11B_S15_R1_001_trimmed_tophat_out.calls.vcf
/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk IndexFeatureFile -F Holly-GC-11-redo_S6_R1_001_trimmed_tophat_out.calls.vcf

Holly_18B_redo_S13_R1_001_trimmed_tophat_out.calls.vcf
Holly_18C_S13_R1_001_trimmed_tophat_out.calls.vcf
Holly-GC-18-redo_S7_R1_001_trimmed_tophat_out.calls.vcf


Holly_1B_S6_R1_001_trimmed_tophat_out.calls.vcf
Holly_1C_S3_R1_001_trimmed_tophat_out.calls.vcf

Holly_21B_S17_R1_001_trimmed_tophat_out.calls.vcf
Holly_21C_S14_R1_001_trimmed_tophat_out.calls.vcf
Holly-GC-21-redo_S8_R1_001_trimmed_tophat_out.calls.vcf


/Users/hollymcqueary/Dropbox/McQueary/Dosage-Compensation/VCFs/gatk IndexFeatureFile -F Holly_2A_S1_R1_001_trimmed_tophat_out.calls.vcf
Holly_2B_S7_R1_001_trimmed_tophat_out.calls.vcf
Holly_2C_S4_R1_001_trimmed_tophat_out.calls.vcf

Holly_31B_S19_R1_001_trimmed_tophat_out.calls.vcf
Holly_31C_S16_R1_001_trimmed_tophat_out.calls.vcf
Holly-GC-31-redo_S9_R1_001_trimmed_tophat_out.calls.vcf


Holly_3A_S2_R1_001_trimmed_tophat_out.calls.vcf
Holly_3B_redo_S12_R1_001_trimmed_tophat_out.calls.vcf
Holly_3C_S5_R1_001_trimmed_tophat_out.calls.vcf

Holly_49B_S20_R1_001_trimmed_tophat_out.calls.vcf
Holly_49C_S17_R1_001_trimmed_tophat_out.calls.vcf
Holly-GC-49-redo_S10_R1_001_trimmed_tophat_out.calls.vcf


Holly_4B_S9_R1_001_trimmed_tophat_out.calls.vcf
Holly_4C_S6_R1_001_trimmed_tophat_out.calls.vcf
Holly-GC-4-redo_S3_R1_001_trimmed_tophat_out.calls.vcf


Holly_59B_S22_R1_001_trimmed_tophat_out.calls.vcf
Holly_59C_S19_R1_001_trimmed_tophat_out.calls.vcf

Holly_5A_S3_R1_001_trimmed_tophat_out.calls.vcf
Holly_5B_S10_R1_001_trimmed_tophat_out.calls.vcf
Holly_5C_redo_S16_R1_001_trimmed_tophat_out.calls.vcf

Holly_61B_S23_R1_001_trimmed_tophat_out.calls.vcf
Holly_61C_S20_R1_001_trimmed_tophat_out.calls.vcf
Holly-GC-61-redo_S11_R1_001_trimmed_tophat_out.calls.vcf


Holly-66B-redo-136232_S14_R1_001_trimmed_tophat_out.calls.vcf
Holly_66C_S21_R1_001_trimmed_tophat_out.calls.vcf

Holly_69B_S25_R1_001_trimmed_tophat_out.calls.vcf
Holly_69C_S22_R1_001_trimmed_tophat_out.calls.vcf

Holly_6A_S4_R1_001_trimmed_tophat_out.calls.vcf
Holly_6B_S11_R1_001_trimmed_tophat_out.calls.vcf
Holly_6C_S8_R1_001_trimmed_tophat_out.calls.vcf

Holly_76B_S26_R1_001_trimmed_tophat_out.calls.vcf
Holly_76C_redo_S18_R1_001_trimmed_tophat_out.calls.vcf

Holly_77B_S27_R1_001_trimmed_tophat_out.calls.vcf
Holly_77C_S24_R1_001_trimmed_tophat_out.calls.vcf

Holly_7B_S12_R1_001_trimmed_tophat_out.calls.vcf
Holly_7C_S9_R1_001_trimmed_tophat_out.calls.vcf
Holly-GC-7-redo_S4_R1_001_trimmed_tophat_out.calls.vcf


Holly_8B_S13_R1_001_trimmed_tophat_out.calls.vcf
Holly_8C_S10_R1_001_trimmed_tophat_out.calls.vcf
Holly-GC-8-redo_S5_R1_001_trimmed_tophat_out.calls.vcf


Holly_9A_S5_R1_001_trimmed_tophat_out.calls.vcf
Holly_9B_S14_R1_001_trimmed_tophat_out.calls.vcf
Holly_9C_S11_R1_001_trimmed_tophat_out.calls.vcf


Holly_GC_Anc_B_S2_R1_001_trimmed_tophat_out.calls.vcf
Holly_GC_Anc_C_S32_R1_001_trimmed_tophat_out.calls.vcf
