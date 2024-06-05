#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=70     # number of CPU per task #4
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=670G   # memory per Nodes   #38
#SBATCH -J "mg"   # job name
#SBATCH --mail-user=carole.belliardo@inrae.fr   # email address
#SBATCH --mail-type=ALL
#SBATCH -e slurm-mg-%j.err
#SBATCH -o slurm-mg-%j.out
#SBATCH -p infinity

module load singularity/3.5.3


### MAIN -------------------------------------------------

SING_IMG='/lerins/hub/projects/25_Metag_PublicData/tools_metagData/Singularity/MetagAssembler.sif'
SING2='singularity exec --bind /bighub/hub:/bighub/hub:rw --bind /lerins/hub:/lerins/hub'

cd '/lerins/hub/projects/25_IPN_Metag/10Metag/0-cat/'

FILENAME='cat_10metag.fasta'
hismR='hifiasm3'

#$SING2 $SING_IMG /lerins/hub/projects/25_Metag_PublicData/tools_metagData/hifiasm-meta/hifiasm_meta -t 70 -S -o ${hismR} $FILENAME > ${hismR}_asm.log

# 2.3.2 --- convert gfa to fasta
# uniquement pour la sortie hifiasm; pas necessaire pour la sortie metaflye

SING_IMG='/lerins/hub/projects/25_Metag_PublicData/tools_metagData/Singularity/pb-metagToolkit2.sif'
SING2='singularity exec  --bind /work/cbelliardo:/work/cbelliardo  --bind /lerins/hub:/lerins/hub'


FILENAME=hifiasm3.p_ctg.gfa
# run en local : 
## gfatools gfa2fa $FILENAME > ${FILENAME}.fasta

#ln -s ${FILENAME}.fasta /lerins/hub/projects/25_IPN_Metag/HiFi-MAG-Pipeline_pacbio/HiFi-MAG-Pipeline_cat/inputs/${FILENAME}.fasta
cd /lerins/hub/projects/25_IPN_Metag/HiFi-MAG-Pipeline_pacbio/HiFi-MAG-Pipeline_cat

$SING2 $SING_IMG /home/tools/conda/bin/snakemake --snakefile Snakefile-hifimags --configfile configs/Sample-Config.yaml -j 70 --use-conda #--conda-frontend conda


