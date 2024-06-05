#!/bin/sh
#$BATCH --workdir /work/gr-fe/kostya/Serena/body/1raw
#SBATCH --job-name=Cram2Vcf
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=50:00:00
#SBATCH --mem=10gb
#SBATCH --output=CramToVcf.%J.out
#SBATCH --error=CramToVcf.%J.err

# Load all modules required
module load gcc/7.4.0
module load samtools/1.9

# Directories, Programs, ResSeq,
OutputDir=/work/gr-fe/kostya/Serena/body/2derived/MtCramMpileupVcf
CramDir=/work/gr-fe/kostya/Serena/body/1raw/Crams2439
export PATH=$PATH:/work/gr-fe/kostya/Serena/body/bcftools-1.9 # export bcftools
RefSeq=/work/gr-fe/kostya/Serena/body/1raw/Hg19MtDna/hs37d5.fa

myfilenames=`ls $CramDir/*/*.cram`  #extansion is  .bam but do not contain 'samtools.' as in /work/gr-fe/archive/sample_repository/BroadNeut/BAMS/Columbia/samtools.586.5433.tmp.0000.bam
for eachfile in $myfilenames
 do
  NameOfFile=$(echo $eachfile | awk '{gsub(/.*\//, "", $0)} 1')  # This should set NameOfFile to the output of awk (filename without the path).
 
  samtools index $eachfile # .cram.crai files are created in each folder

  samtools idxstats $eachfile  > $OutputDir/$NameOfFile.idxstats
  samtools flagstat $eachfile  > $OutputDir/$NameOfFile.flagstat

  samtools view -M -f 0x40 -F 0x4 $eachfile MT -o $OutputDir/$NameOfFile.MT1.cram
  samtools view -M -f 0x80 -F 0x4 $eachfile MT -o $OutputDir/$NameOfFile.MT2.cram
  samtools view -M -F 0x4 $eachfile MT -o $OutputDir/$NameOfFile.MT.cram

  samtools mpileup -f $RefSeq $OutputDir/$NameOfFile.MT1.cram -o $OutputDir/$NameOfFile.MT1.cram.mpileup
  samtools mpileup -f $RefSeq $OutputDir/$NameOfFile.MT2.cram -o $OutputDir/$NameOfFile.MT2.cram.mpileup
  samtools mpileup -f $RefSeq $OutputDir/$NameOfFile.MT.cram -o $OutputDir/$NameOfFile.MT.cram.mpileup

  bcftools mpileup -f $RefSeq $OutputDir/$NameOfFile.MT.cram | bcftools view -o $OutputDir/$NameOfFile.MT.cram.vcf
done

# samtools view -M -f 0x40 -F 0x4 $CramDir/22101_1#24.cram MT -o $OutputDir/MT1.sam
# samtools view -M -f 0x80 -F 0x4 $CramDir/22101_1#24.cram MT -o $OutputDir/MT2.sam

# samtools view -M -F 0x4 $CramDir/22101_1#24.cram MT | bcftools view -o $OutputDir/test.vcf


# bcftools view $OutputDir/MT1.sam -o $OutputDir/test.vcf

##### WORKING LINES:

# samtools view -M -F 0x4 $CramDir/22101_1#24.cram MT -o $OutputDir/MT.cram  # here I need to add more correct filters (think!!!) 
# samtools mpileup -f $RefSeq $OutputDir/MT.cram -o $OutputDir/MT.mpileup 
# bcftools mpileup -f $RefSeq $OutputDir/MT.cram | bcftools view -o $OutputDir/MT.vcf

# these two lines work one after another - but if I add pipe - some problems - Failed to open -: unknown file type

### understanding, downloading and indexing of the correct refseq hs37d5.fa
# samtools view -H $OutputDir/MT.cram > $OutputDir/MT.cram.h # копаемся в хедерах, видим что референс один и называется hs37d5.fa
# cd /work/gr-fe/kostya/Serena/body/1raw/Hg19MtDna
# wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz #референс с таким названием и единым файлом используется в 1000G
# gunzip -d hs37d5.fa.gz
# samtools faidx hs37d5.fa

# samtools mpileup -f hs37d5.fa MT.cram -o MT.mpileup #MT.mpileup в приложении к письму


# cd /work/gr-fe/kostya/Serena/body/1raw/Hg19MtDna
# wget  http://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chrM.fa.gz


#samtools mpileup $CramDir/MT1.view -o $CramDir/MT1.mpileup
#samtools mpileup $CramDir/MT2.view -o $CramDir/MT2.mpileup


## RUN: sbatch 
## HEADER: script walks along several directories with bam files obtained after Exome Sequencing and extracts info about mtDNA reads
## many bam files belong to different exome capture kids and obtained / mapped with different pipelines and thus this test is pilot - to see abundance of mtDNA reads
## also this script is naive and consequitive (not parallel) and it takes ~ 50 hours to run
## INPUTS: bam files in many directories inside /work/gr-fe/archive/sample_repository/
## OUTPUTS: three types of iles (idxstats, flagstat, depth) for each bam file in the directory: /work/gr-fe/kostya/MtDnaFromExomeData/body/2Derived/MtDnaStat
# cd /work/gr-fe/kostya/MtDnaFromExomeData/body/2Derived/MtDnaStat
# ls | wc -l # 3069

## Transfer data to my laptop (to work with R)
## cd  # go home at laptop
## scp -r popadin@helvetios.epfl.ch:/work/gr-fe/kostya/MtDnaFromExomeData/body/2Derived/MtDnaStat .  # it will copie the whole folder MtDnaStat
## Rstudio: MtDnaStatAnalysis.01.R, MtDnaStatAnalysis.02.R

#Sinteract -n 16 -t 01:00:00 -m 32G -p debug
#$ module load gcc/7.4.0
#$ module load samtools/1.9
#$ samtools view 22101_1#24.cram 1 > -o out.txt
#[main_samview] random alignment retrieval only works for indexed BAM or CRAM files.

# vim MT.txt
# explained on https://en.wikipedia.org/wiki/SAM_(file_format)
# column 5: MAPQ	Int	MApping Quality: -10*log10(probability that mapping position is wrong): the higher the value the better, but if it is 255 - p value unavilaible. 

# samtools mpileup 22101_1#24.cram -r MT -o MT.pileup

##### 
#Костя, вот пороги для того чтобы МЯГКО экстрагировать данные по риду 1:
#samtools view -M -f 0x40 -F 0x4 -u ../hg19.bam $chrPoint 2>/dev/null | samtools mpileup -A --ff 4 -B -x - 2>/dev/null
#для того чтобы МЯГКО экстрагировать данные по риду 2:  
#samtools view -M -f 0x80 -F 0x4 -u ../hg19.bam $chrPoint 2>/dev/null | samtools mpileup -A --ff 4 -B -x - 2>/dev/null

#view  
#-M Use the multi-region iterator on the union of the BED file and command-line region arguments. This avoids re-reading the same regions of files so can sometimes be much faster. Note this also removes duplicate sequences. Without this a sequence that overlaps multiple regions specified on the command line will be reported multiple times. I didn't note effect of this file on mtDNA - probably - because it is impossible, that within mtDNA (within this regions) a read can be mapped to different locations.
#-u Output uncompressed BAM. This option saves time spent on compression/decompression and is thus preferred when the output is piped to another samtools command.
#-F Do not output alignments with any bits set in INT present in the FLAG field. INT can be specified in hex by beginning with `0x' (i.e. /^0x[0-9A-F]+/) or in octal by beginning with `0' (i.e. /^0[0-7]+/) [0].
#-F 0x4 NOT unmapped reads https://broadinstitute.github.io/picard/explain-flags.html
#-f Only output alignments with all bits set in INT present in the FLAG field. INT can be specified in hex by beginning with `0x' (i.e. /^0x[0-9A-F]+/) or in octal by beginning with `0' (i.e. /^0[0-7]+/) [0].
#-f 0x40 / 0x80  first / second read in pair https://broadinstitute.github.io/picard/explain-flags.html

#mpileup 
#-A  Do not skip anomalous read pairs in variant calling.
#--ff 4  Filter flags: skip reads with mask bits set [UNMAP] // я использую этот дубль т.к. view -F 0x4 иногда может не срабатывать (по неизвестным причинам)
#-B Disable base alignment quality (BAQ) computation. // ускоряет анализ, обычно не нужен
#-x Disable read-pair overlap detection. // ускоряет анализ

#samtools view -M -f 0x40 -F 0x4 22101_1#24.cram MT -o MT1.txt
#samtools view -M -f 0x80 -F 0x4 22101_1#24.cram MT -o MT2.txt
#samtools mpileup 22101_1#24.cram -O --s -r MT -o MT.pileup
#samtools mpileup -uf ref.fa all-aln.bam | bcftools view -vcs xxx -1 yyy - > out.vcf amtools view -M -F 0x4 $CramDir/22101_1#24.cram MT -o MT.cram

