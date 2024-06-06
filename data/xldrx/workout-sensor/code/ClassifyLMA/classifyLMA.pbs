#!/bin/bash

#PBS -l walltime=04:00:00
#PBS -l nodes=1:ppn=12
#PBS -q secondary 
#PBS -j oe



unset DISPLAY

module load matlab

export IFS=“-“

i=1

for word in ${PBS_JOBNAME}; do
   arr[i]=$word
   ((i++))
done

cd $PBS_O_WORKDIR



matlab -r runClassifier\(${arr[2]},${arr[3]}\)
