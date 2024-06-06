#!/bin/bash
#SBATCH --job-name=3.3_LK_SB_1p.sh                                               #Job name
#SBATCH --partition=batch		                                                    #Partition (queue) name
#SBATCH --ntasks=4			                                                        #Single task job
#SBATCH --cpus-per-task=12                                                      #Number of cores per task
#SBATCH --mem=24gb			                                                        #Total memory for job
#SBATCH --time=72:00:00  		                                                    #Time limit hrs:min:sec
#SBATCH --output=/scratch/ahw22099/FireAnt_GRN/std_out/3.3_LK_SB_1p.log.%j			#Standard output
#SBATCH --error=/scratch/ahw22099/FireAnt_GRN/std_out/3.3_LK_SB_1p.err.%j		    #Standard error log
#SBATCH --mail-user=ahw22099@uga.edu                                            #Where to send mail -
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS                                        #Mail events (BEGIN, END, FAIL, ALL)
#SBATCH --array=0-52

################## STAR ##################
module load STAR/2.7.10b-GCC-11.3.0
########### SB ############

LK_trimmed_fq="/scratch/ahw22099/FireAnt_GRN/LK_trimmed_fq"
if [ ! -d $LK_trimmed_fq ]
then
mkdir -p $LK_trimmed_fq
fi

STAR_genome_SB="/scratch/ahw22099/FireAnt_GRN/STAR_genome_SB"
if [ ! -d $STAR_genome_SB ]
then
mkdir -p $STAR_genome_SB
fi


SB_genome="/scratch/ahw22099/FireAnt_GRN/UNIL_Sinv_3.4_SB"
if [ ! -d $SB_genome ]
then
mkdir -p $SB_genome
fi

##1st pass
LK_STAR_SB="/scratch/ahw22099/FireAnt_GRN/LK_STAR_SB"
if [ ! -d $LK_STAR_SB ]
then
mkdir -p $LK_STAR_SB
fi

FirstPass_SB="/scratch/ahw22099/FireAnt_GRN/LK_STAR_SB/1p_out"
if [ ! -d $FirstPass_SB ]
then
mkdir -p $FirstPass_SB
fi

cd $LK_trimmed_fq
#make sample list for array job
R1_sample_list=($(<LK_trimmed_input_list_1.txt))
R2_sample_list=($(<LK_trimmed_input_list_2.txt))
R1=${R1_sample_list[${SLURM_ARRAY_TASK_ID}]}
R2=${R2_sample_list[${SLURM_ARRAY_TASK_ID}]}

echo $R1
echo $R2

base=`basename "$R1" .R1_val_1.fq.gz`

STAR \
--readFilesCommand zcat \
--runThreadN 8 \
--genomeDir $STAR_genome_SB \
--readFilesIn $R1 $R2 \
--outFilterType BySJout \
--outFilterMultimapNmax 20 \
--alignSJoverhangMin 8 \
--alignSJDBoverhangMin 1 \
--outFilterMismatchNoverLmax 0.05 \
--alignIntronMin 20 \
--alignIntronMax 1000000 \
--genomeLoad NoSharedMemory \
--outSAMtype BAM SortedByCoordinate \
--outSAMstrandField intronMotif \
--outSAMattrIHstart 0 \
--outFileNamePrefix $FirstPass_SB/"$base".SB1pass. \
--limitBAMsortRAM 30000000000
#
