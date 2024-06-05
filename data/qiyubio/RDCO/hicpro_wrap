#!/bin/bash

## Author: Qi Yu
## Contact: dryuqi@gmail.com

Soft="hicpro_wrap"
#VERSION="1.0.0"
#this is a parallel module for HiCpro 2.11.1 runnable under biowulf

function usage {
	echo -e "usage : $Soft -i FASTQ -j FASTQ -o OUTPUT -g GENOME"
}

function help {
    usage;
    echo 
    echo "$Soft"
    echo "---------------"
    echo "OPTIONS"
    echo
    echo "   -i FASTQ : input _R1.fastq or fastq.gz file"
    echo "   -j FASTQ : input _R2.fastq or fastq.gz file"
    echo "   -o OUTPUT : physical path you want output locate, where to generate align folder"
    echo "   -g GENOME : genome mm10_B6 or mm10_CASTXB6 "
    echo "   [-p] : if specified run $Soft on a cluster"
    echo "   [-h]: help"
   # echo "   [-v|--version]: version"
    exit;
}

function opts_error {
    echo -e "Error : invalid parameters !" >&2
    echo -e "Use $Soft -h for help"
    exit
}

if [ $# -lt 1 ]
then
    usage
    exit
fi


CLUSTER=0
#MAKE_OPTS=""
INPUT1=""
INPUT2=""
OUTPUT=""
GENOME=""



while getopts ":i:j:o:g:h" OPT
do
    case $OPT in
	i) INPUT1=$OPTARG;;
	j) INPUT2=$OPTARG;;
	o) OUTPUT=$OPTARG;;
	g) GENOME=$OPTARG;;
	#s) MAKE_OPTS="$MAKE_OPTS $OPTARG";;
	p) CLUSTER=1 ;;
	h) help ;;
	\?)
	     echo "Invalid option: -$OPTARG" >&2
	     usage
	     exit 1
	     ;;
	 :)
	     echo "Option -$OPTARG requires an argument." >&2
	     usage
	     exit 1
	     ;;
    esac
done


#if [[ -z $INPUT1 || -z $INPUT2  || -z $OUTPUT  || -g $GENOME ]]; then
#    usage
#    exit
#fi

###################################
#check
##################################

if [ ! -e $INPUT1 ]; then
    echo "Inputs '$INPUT1' not found. Exit."
    exit -1
fi
if [ ! -e $INPUT2 ]; then
    echo "Inputs '$INPUT2' not found. Exit."
    exit -1
fi


##############################
#clean
###################################

module load hicpro;
#genom=$2
#enzyme=$3
#fastq=$1
#sampleid=$4
echo "Your fastq will be splited at $OUTPUT"



nbin1=$(find -L $INPUT1 -name "*.fastq" -o -name "*.fastq.gz" | wc -l)
nbin2=$(find -L $INPUT2 -name "*.fastq" -o -name "*.fastq.gz" | wc -l)

if [ $nbin1 == 0 ]; then
	die "Error: Directory Hierarchy of rawdata '$INPUT1' is not correct. No '.fastq(.gz)' files detected"
fi

if [ $nbin2 == 0 ]; then
        die "Error: Directory Hierarchy of rawdata '$INPUT2' is not correct. No '.fastq(.gz)' files detected"
fi

#############################
#convert genome to config file
###################################

if [[ $GENOME == 'mm10_B6' ]] ; then
    CONFIGfile='/data/chengg2/config_mouse_biowulf_B6.txt'

elif [[ $GENOME == 'mm10_CASTXB6' ]] ; then
    CONFIGfile='/data/chengg2/config_mouse_biowulf.txt'
else 
    echo "unrecognized genome"
fi

echo "The config file you are going to use is $CONFIGfile, please check"

#################################
#run program
#################################

if [ -d $OUTPUT/$OUTPUT ]; 

then echo "file has been splited or you should remove $OUTPUT/$OUTPUT and try again" 

else

mkdir $OUTPUT
mkdir $OUTPUT/$OUTPUT; 



echo -n "Split the R1 file, this might take a while"

/usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/bin/utils/split_reads.py -r $OUTPUT/$OUTPUT -n 5000000 $INPUT1
echo -e "\nThe R1 file has been splited\n"


echo -n "Split the R2 file, this might take a while"


/usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/bin/utils/split_reads.py -r $OUTPUT/$OUTPUT -n 5000000 $INPUT2 

echo -e "\nThe R2 file has been splited\n"

fi
#This program has been only tested for hicpro2.11.1 edited by Qi and run at biowulf, please edited accordingly

echo "Your output will be at ${OUTPUT}_hicpro_result"

HiC-Pro -i $OUTPUT -o ${OUTPUT}_hicpro_result -c ${CONFIGfile} -p

CLUSTER=$(wc -l < ${OUTPUT}_hicpro_result/inputfiles_IMR90_split.txt)

cd ${OUTPUT}_hicpro_result

rm HiCPro_step*_IMR90_split.sh

awk -v var=$CLUSTER -v varb=$CONFIGfile '''BEGIN {print ("#!/bin/bash\n#SBATCH --ntasks=1\n#SBATCH --mem=40g\n#SBATCH --time=6:00:00\n#SBATCH --cpus-per-task=24\n#SBATCH --job-name=HiCpro_split\n#SBATCH --array=1-" var "\nFASTQFILE=$SLURM_SUBMIT_DIR/inputfiles_IMR90_split.txt; export FASTQFILE\nmake --file /usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/scripts/Makefile CONFIG_FILE=" varb " CONFIG_SYS=/usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/config-system.txt mapping  2>&1")}''' >HiCPro_step1_split.sh

awk -v varb=$CONFIGfile '''BEGIN {print ("#!/bin/bash\n#SBATCH --ntasks=1\n#SBATCH --mem=40g\n#SBATCH --time=2:00:00\n#SBATCH --cpus-per-task=24\n#SBATCH --job-name=HiCpro_check\ncd $SLURM_SUBMIT_DIR\nmodule load R\nmake --file /usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/scripts/Makefile CONFIG_FILE=" varb " CONFIG_SYS=/usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/config-system.txt quality_checks  2>&1")}''' >HiCPro_step2_split.sh

awk -v var=$CLUSTER -v varb=$CONFIGfile '''BEGIN {print ("#!/bin/bash\n#SBATCH --ntasks=1\n#SBATCH --mem=40g\n#SBATCH --time=8:00:00\n#SBATCH --cpus-per-task=24\n#SBATCH --job-name=HiCpro_matrix\n#SBATCH --array=1-" var "\nFASTQFILE=$SLURM_SUBMIT_DIR/inputfiles_IMR90_split.txt; export FASTQFILE\nmake --file /usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/scripts/Makefile CONFIG_FILE=" varb " CONFIG_SYS=/usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/config-system.txt proc_hic  2>&1")}''' >HiCPro_step3_split.sh

awk -v varb=$CONFIGfile '''BEGIN {print ("#!/bin/bash\n#SBATCH --ntasks=1\n#SBATCH --mem=80g\n#SBATCH --time=8:00:00\n#SBATCH --cpus-per-task=24\n#SBATCH --job-name=HiCpro_merge\ncd $SLURM_SUBMIT_DIR\nmodule load R\nmake --file /usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/scripts/Makefile CONFIG_FILE=" varb " CONFIG_SYS=/usr/local/Anaconda/envs_app/hicpro/2.11.1/HiC-Pro_2.11.1/config-system.txt merge_persample 2>&1")}''' >HiCPro_step4_split.sh

jid1=$(sbatch HiCPro_step1_split.sh)
jid2=$(sbatch --dependency=afterok:$jid1 HiCPro_step2_split.sh)
jid3=$(sbatch --dependency=afterany:$jid2 HiCPro_step3_split.sh)
jid4=$(sbatch --dependency=afterany:$jid3 HiCPro_step4_split.sh)

echo "All jobs have been submitted"

#################################################


