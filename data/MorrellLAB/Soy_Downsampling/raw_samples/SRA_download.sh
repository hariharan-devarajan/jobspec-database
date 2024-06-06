#!/bin/env sh

#PBS -l mem=12gb,nodes=1:ppn=1,walltime=24:00:00 
#PBS -m abe 
#PBS -M wyant008@umn.edu 
#PBS -q lab
#PBS -e /panfs/roc/groups/9/morrellp/shared/Projects/Soy_Downsampling/Raw_Samples
#PBS -o /panfs/roc/groups/9/morrellp/shared/Projects/Soy_Downsampling/Raw_Samples


#    Collect file from command line
SRA_FILES=/panfs/roc/groups/9/morrellp/shared/Projects/Soy_Downsampling/Raw_Samples/soy_run_numbers.txt

#   Make sure the file exists
if ! [[ -f "${SRA_FILES}" ]]
    then echo "Failed to find ${SRA_FILES}, exiting..." >&2
    exit 1
    fi 

#   Make the array using command substitution
declare -a SRA_ARRAY=($(cat "${SRA_FILES}")) 

#   Print the values of the array to screen
printf '%s\n' "${SRA_ARRAY[@]}"

#   location of SRA download script, from Tom Kono's Misc_Utils
SRA_FETCH=/panfs/roc/groups/9/morrellp/shared/Projects/Soy_Downsampling/Raw_Samples/SRA_Fetch.sh

#   directory where SRA files will be downloaded
OUTPUT=/panfs/roc/scratch/wyant008/Soy_Downsampling/Raw_Samples

#   iterate over every each of the run numbers in a lit of SRA files
#   and download to specified directory
#   in SRA_Fetch -r = run #, -e experiment #, -p sample #, -s study #
for i in "${SRA_ARRAY[@]}"
do 
	bash $SRA_FETCH -r $i -d $OUTPUT
	sleep 5m
done

