#!/bin/bash
#SBATCH --job-name=4.3_LK_SB_2p.sh                                               #Job name
#SBATCH --partition=batch		                                                    #Partition (queue) name
#SBATCH --ntasks=1			                                                        #Single task job
#SBATCH --cpus-per-task=10                                                      #Number of cores per task
#SBATCH --mem=24gb			                                                        #Total memory for job
#SBATCH --time=72:00:00  		                                                    #Time limit hrs:min:sec
#SBATCH --output=/scratch/ahw22099/FireAnt_GRN/std_out/4.3_LK_SB_2p.log.%j			#Standard output
#SBATCH --error=/scratch/ahw22099/FireAnt_GRN/std_out/4.3_LK_SB_2p.err.%j		    #Standard error log
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

SecondPass_SB="/scratch/ahw22099/FireAnt_GRN/LK_STAR_SB/2p_out"
if [ ! -d $SecondPass_SB ]
then
mkdir -p $SecondPass_SB
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
--quantMode GeneCounts \
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
--outFileNamePrefix $SecondPass_SB/"$base".SB2pass. \
--limitBAMsortRAM 30000000000 \
--sjdbFileChrStartEnd $FirstPass_SB/B_mBB_1.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mBB_2.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mBB_3.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mBB_4.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mBB_5_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mBB_6.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mBB_6_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mBB_7.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mpBB_10.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mpBB_11_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mpBB_12.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mpBB_13_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mpBB_14_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mpBB_15.SB1pass.SJ.out.tab \
$FirstPass_SB/B_mpBB_16.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBB_1.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBB_2.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBB_3.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBB_4.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBB_5_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBB_6_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBB_7_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBL_1.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBL_2.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBL_3.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBL_4.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBL_5.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBL_6_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/B_pBL_7_merge.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mBB_1.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mBB_3.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mBB_5.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mBB_6.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mBB_7.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mBB_8.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mpBB_10.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mpBB_11.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mpBB_13.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mpBB_14.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mpBB_15.SB1pass.SJ.out.tab \
$FirstPass_SB/G_mpBB_17.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBB_1.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBB_3.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBB_5.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBB_6.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBB_7.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBB_8.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBL_1.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBL_3.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBL_5.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBL_6.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBL_7.SB1pass.SJ.out.tab \
$FirstPass_SB/G_pBL_8.SB1pass.SJ.out.tab
