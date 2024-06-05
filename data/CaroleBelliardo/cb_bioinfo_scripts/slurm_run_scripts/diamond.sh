#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --cpus-per-task=30     # number of CPU per task #4
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=250G   # memory per Nodes   #38
#SBATCH -J "dmd"   # job name
#SBATCH --mail-user=carole.belliardo@inrae.fr   # email address
#SBATCH --mail-type=ALL
#SBATCH -e slurm-dmd-%j.err
#SBATCH -o slurm-dmd-%j.out
#SBATCH -p all


module load singularity/3.5.3

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

## -- makedb
SING_IMG='/database/hub/SINGULARITY_GALAXY/diamond:2.1.7--h5b5514e_0'
SING2='singularity exec --bind /kwak/hub/25_cbelliardo/MetaNema_LRmg/10Metag'

db_fasta=$1ls
db=$(echo $db_fasta |cut-d '.' -f1)'.dmnd'

$SING2 $SING_IMG diamond makedb -p $SLURM_JOB_CPUS_PER_NODE --in $db_fasta --db $db 
echo 'make db ok'

## -- Run
#QUERY=$2
#OUT=$(echo $QUERY |cut -d '.' -f1)'__vs__'$(echo $db_fasta |cut-d '.' -f1)'.dmnd.out'

#Run a search in blastp mode
#$SING2 $SING_IMG diamond blastp --sensitive -k500000000000 --id 99 -p $SLURM_JOB_CPUS_PER_NODE --outfmt 6 -d $db -q $QUERY -o $OUT 

## run with : sbatch diamond.sh <db.fasta> <query.fasta>

# TODO : run on /kwak/hub/25_cbelliardo/MetaNema_LRmg/10Metag/Proteins_asm_mf_reads.aa