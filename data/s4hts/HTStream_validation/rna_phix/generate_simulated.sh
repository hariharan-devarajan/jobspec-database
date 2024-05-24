#!/bin/bash

#SBATCH --job-name=simulate_data # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --time=900
#SBATCH --mem=9000 # Memory pool for all cores (see also --mem-per-cpu)
##SBATCH --partition=production
##SBATCH --reservation=workshop
##SBATCH --account=workshop
##SBATCH --array=1-3
#SBATCH --output=slurmout/simulate_data_%A_%a.out # File to which STDOUT will be written
#SBATCH --error=slurmout/simulate_data_%A_%a.err # File to which STDERR will be written

start=`date +%s`
echo $HOSTNAME
#echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

#sample=`sed "${SLURM_ARRAY_TASK_ID}q;d" ${2}`
#type=$1
#sample_nosuffix=${sample}
#sample=${sample}_${type}
#inpath='/share/biocore/keith/benchmarking/ena' 
#outpath='01-HTS_Preproc'
#[[ -d ${outpath} ]] || mkdir ${outpath}
#[[ -d ${outpath}/${sample} ]] || mkdir ${outpath}/${sample}

#echo "SAMPLE: ${sample}"
#echo "SAMPLE NO SUFFIX: ${sample_nosuffix}"
#module load htstream/1.3.2



#if [[ ${type} == "seqscreener_phix" ]]; then

length=250
bbmap_dir='/share/biocore/keith/benchmarking/tools/bbmap'
reference='References/GRCh38.primary_assembly.genome.fa'
outdir='01-GoldStandard/human_genome/'
${bbmap_dir}/./randomreads.sh ref=${reference} out1='${outdir}/human_genome1.fastq.gz' out2='${outdir}/human_genome2.fastq.gz' length=$length reads=500000 paired=t


end=`date +%s`
runtime=$((end-start))
echo $runtime
