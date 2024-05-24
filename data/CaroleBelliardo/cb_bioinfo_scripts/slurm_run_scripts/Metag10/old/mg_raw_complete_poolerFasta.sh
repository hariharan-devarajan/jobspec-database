#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=2     # number of CPU per task #4
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per Nodes   #38
#SBATCH -J "mg"   # job name
#SBATCH --mail-user=carole.belliardo@inrae.fr   # email address
#SBATCH --mail-type=ALL
#SBATCH -e slurm-mg-%j.err
#SBATCH -o slurm-mg-%j.out
#SBATCH -p all

module load singularity/3.5.3


### MAIN -------------------------------------------------

#SING_IMG='/lerins/hub/projects/25_Metag_PublicData/tools_metagData/Singularity/MetagAssembler.sif'
#SING2='singularity exec --bind /work/cbelliardo:/work/cbelliardo --bind /bighub/hub:/bighub/hub:rw --bind /lerins/hub:/lerins/hub'

#wd='/lerins/hub/projects/25_IPN_Metag/10Metag/fastq'
#cd $wd

#FILES=($(ls -1 *.fastq))
#FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}
#PREFS=($(ls -1 *.fastq | cut -f1 -d'.'))
#PREF=${FILES[$SLURM_ARRAY_TASK_ID]}


## 1. --- convert fastq to fasta
#outR='/lerins/hub/projects/25_IPN_Metag/10Metag/fasta'
#fasta=${outR}/${PREF}.fasta
#$SING2 $SING_IMG python /lerins/hub/projects/25_20191015_git/python/FASTQ_to_FASTA/FASTQ_to_FASTA.py $FILENAME $fasta


## 2. --- assemble fasta with metaflye & hifiasm
#cd $outR

#FILES=($(ls -1 *.fasta))
#FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}
#PREFS=($(ls -1 *.fasta | cut -f1 -d'.'))
#PREF=${PREFS[$SLURM_ARRAY_TASK_ID]}


# 2.1-- METAFLYE default parameters
#flyR='/lerins/hub/projects/25_IPN_Metag/10Metag/assembly/metafly/'${PREF}'/'
#mkdir $flyR
#$SING2 $SING_IMG flye --pacbio-hifi $FILENAME --out-dir ${flyR} -t $SLURM_CPUS_PER_TASK  --meta


# 2.2-- HIFIasm default parameters
## 2.2.1 --- run hifiasm tool
#hismR='/lerins/hub/projects/25_IPN_Metag/10Metag/assembly/hifiasm/'${PREF}'/'
#mkdir $hismR
#$SING2 $SING_IMG /lerins/hub/projects/25_Metag_PublicData/tools_metagData/hifiasm-meta/hifiasm_meta -t $SLURM_CPUS_PER_TASK  --force-preovec -o ${hismR}/${PREF} $FILENAME > ${hismR}/asm.log


## 2.2.2 --- convert gfa to fasta
## uniquement pour la sortie hifiasm; pas necessaire pour la sortie metaflye
#cd $hismR
#FILES=($(ls -1 */*.p_ctg.gfa))
#FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}

#$SING2 $SING_IMG gfatools gfa2fa $FILENAME > ${FILENAME}.fasta


# 2.3-- HIFIasm -S parameters
## 2.3.1 --- run hifiasm tool
#hismR='/lerins/hub/projects/25_IPN_Metag/10Metag/assembly/hifiasm3/'${PREF}'/'
#mkdir $hismR
#$SING2 $SING_IMG /lerins/hub/projects/25_Metag_PublicData/tools_metagData/hifiasm-meta/hifiasm_meta -t 70 -S -o ${hismR}/${PREF} $FILENAME > ${hismR}/asm.log

## 2.3.2 --- convert gfa to fasta
## uniquement pour la sortie hifiasm; pas necessaire pour la sortie metaflye
#cd $hismR
#FILES=($(ls -1 */*.p_ctg.gfa))
#FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}

#$SING2 $SING_IMG gfatools gfa2fa $FILENAME > ${FILENAME}.fasta



## 3. stats fasta files
# 3.1-- metaflye output
#cd $flyR
#FILES=($(ls -1 *.fasta))
#FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}

#$SING2 $SING_IMG seqkit stats -j 70 $FILENAME >> stat_seqkit.txt

# 3.1-- hifiasm output
#cd $hismR
#FILES=($(ls -1 *.fasta))
#FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}

#$SING2 $SING_IMG seqkit stats -j 70 $FILENAME >> stat_seqkit.txt


## 4. Compare assemblages
# 4.1-- dotplot
# 4.1.1 D-Genies
# 4.1.1 mashmap

# 4.2-- pacbio workflow
#SING_IMG='/lerins/hub/projects/25_Metag_PublicData/tools_metagData/Singularity/pb-metagToolkit2.sif'
#SING2='singularity exec  --bind /work/cbelliardo:/work/cbelliardo  --bind /lerins/hub:/lerins/hub'

#cd /lerins/hub/projects/25_IPN_Metag/HiFi-MAG-Pipeline ## doit contenir les reads + contigs dans repo 'inputs' + modifier les fichiers configs.yalm + configs/config...

## run tests : 
#$SING2 $SING_IMG /home/tools/conda/bin/snakemake  -np --snakefile Snakefile-hifimags --configfile configs/Sample-Config.yaml
#$SING2 $SING_IMG /home/tools/conda/bin/snakemake --dag --snakefile Snakefile-hifimags --configfile configs/Sample-Config.yaml | dot -Tsvg > /lerins/hub/tmp/hifimags_analysis.svg

# run on samples: 
#$SING2 $SING_IMG /home/tools/conda/bin/snakemake --snakefile Snakefile-hifimags --configfile configs/Sample-Config.yaml -j 60 --use-conda #--conda-frontend conda


#**********************************************************************************************************************************************************************************************************

## 5. -- compo reads
#SING_IMG='/lerins/hub/projects/25_Metag_PublicData/tools_metagData/Singularity/MetagAssembler.sif'
#cd $outR

#FILES=($(ls -1 *.fasta))
#FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}
#PREFS=($(ls -1 *.fasta | cut -f1 -d'.'))
#PREF=${FILES[$SLURM_ARRAY_TASK_ID]}

#db_kraken='/lerins/hub/DB/RefSeq_genomic/RefSeq_genomic_kraken'

#SING_IMG='/lerins/hub/projects/25_Metag_PublicData/tools_metagData/Singularity/MetagTools3.8.sif'
#SING2='singularity exec --bind /work/cbelliardo:/work/cbelliardo --bind /bighub/hub:/bighub/hub:rw --bind /lerins/hub:/lerins/hub'

#out='/lerins/hub/projects/25_IPN_Metag/10Metag/kraken_reads'

#$SING2 $SING_IMG /home/tools/kraken/kraken2 --confidence 0.03 --threads 60 --db $db_kraken $FILENAME --output ${out}/${PREF}.krak


## 6.-- prediction prot
# 6.1-- prodigal
#wd='/lerins/hub/projects/25_IPN_Metag/prodigal/'

#$SING2 $SING_IMG prodigal -f gff -p anon -i $rawFasta -o ${wd}/rawP -a ${wd}/rawP_prot -s ${wd}/raw_startCompl 
#$SING2 $SING_IMG prodigal -f gff -p anon -i $assembly -o ${wd}/assembly -a ${wd}/ass_prot -s ${wd}/ass_startCompl -w ${wd}/ass_stat

# 6.2-- Metaeuk



