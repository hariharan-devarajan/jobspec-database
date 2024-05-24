#!/bin/sh

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -p defq
#SBATCH --job-name=abra

### This is a batch array script. To run it use sbatch --array [0-n-1] <SCRIPT>, where n = # of input bam files
### This will run realignment of each bam file separately in parallel. 

cd $SLURM_SUBMIT_DIR


#source appropriate environments to enable use of conda installs through batch submission
source /home/lw74/.bashrc
source /home/lw74/.bash_profile
conda activate abra_env

# Load necessary modules
module load samtools
module load java


# Set up input parameters
refGenome="/work/lw74/refGenome/Pbar.2022.LG.fa"
#abra2="/work/lw74/software_installs/abra2/abra2-2.24.jar"
threads=4
outdir="/work/lw74/habro/mapped_filtered_realigned_bams"
mkdir -p "/work/lw74/habro/mapped_filtered_realigned_bams"


# Make array for specifying the input and output files
bamdir="/work/lw74/habro/mapped_filtered_bams"
flist=$(ls $bamdir/*mapped_filtered.bam)
flist=(${flist[@]})
bfile="${flist[$SLURM_ARRAY_TASK_ID]}"
logname="$(basename $bfile | sed 's/_mapped_filtered.bam/.abra.realign.log/g')"
rfname="$(basename $bfile | sed 's/.bam/.realigned.bam/g')"


## Realign all bam files
#java -Xmx16G -jar $abra2 --in $bfile --out $outdir/$rfname --ref $refGenome --threads $threads --tmpdir . > $logname
abra2 --in $bfile --out $outdir/$rfname --ref $refGenome --threads $threads --tmpdir . > $logname


## Index the realigned bam files
samtools index $outdir/$rfname $outdir/$rfname".bai"
