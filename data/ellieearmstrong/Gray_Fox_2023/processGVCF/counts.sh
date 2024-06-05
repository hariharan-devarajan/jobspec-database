#!/bin/sh
#SBATCH --job-name=countGVCF
#SBATCH --output=/scratch/users/elliea/jazlyn-ellie/grayfox_2023/count.out  #you will need to modify the path
#SBATCH --error=/scratch/users/elliea/jazlyn-ellie/grayfox_2023/count.err  #you will need to modify the path
#SBATCH --time=02:00:00 #ten hour run time
#SBATCH -p normal #the main partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1 #one cpu per task
#SBATCH --mem-per-cpu=500MB #this is equivalent to 10G
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=jaam@stanford.edu


for f in {1..32}
do

#zgrep -v "#"  chrom"$f"_allFoxes.rg.md.haplotypecaller.all.g.renameChroms.gvcf.gz | wc -l >> numSitesPrefilter.out

#zgrep -v "#"  chrom"$f"_allFoxes.rg.md.haplotypecaller.all.g.renameChroms.mappabilityFilter.gvcf.gz | wc -l >> numSitesPostfilter.out

#wc -l chrom"$f"_multiAllelic.txt >> numSitesMuliAllelicPostfilter.out 
#zgrep -v "#" chrom"$f"_allFoxes.rg.md.haplotypecaller.all.g.renameChroms.mappabilityFilter.AN.QUAL.DP.biallelic.gvcf.gz | wc -l >> numBiallelicSitesPostfilter.out

#echo "chrom $f"

done
