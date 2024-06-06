#!/bin/bash

#PBS -N project_report
#PBS -l walltime=01:00:00
#PBS -l mem=1GB
#PBS -l nodes=1:ppn=8
#PBS -S /bin/bash
#PBS -j oe

# Change to directory from where PBS job was submitted
cd $PBS_O_WORKDIR

# export OMP_NUM_THREADS=$PBS_NP

# Compile the program with compiler version 2014 and -O3 optimizer.
c++ -g -Wall -fopenmp -lpng -std=c++14 -O3 PNG.cpp ImageSearch.cpp -o ImageSearch

# Loop for running the program using different threads.
for threads in 1 4 8;
do
    echo "---------------------[ Threads: ${threads} ]---------------------"
    export OMP_NUM_THREADS=${threads}
    for n in 1 2 3 4 5;
    do
        echo "----------------[ Cancer_mask.png: ${threads} ]---------------------"
        /usr/bin/time -v ./ImageSearch Mammogram.png Cancer_mask.png result_Mammogram_${threads}.png true 75 32
        echo "-------------------[ and_mask.png: ${threads} ]---------------------"
        /usr/bin/time -v ./ImageSearch TestImage.png and_mask.png result_TestImage_${threads}.png true 75 16
        echo "---------------[ WindowPane_mask.png: ${threads} ]------------------"
        /usr/bin/time -v ./ImageSearch MiamiMarcumCenter.png WindowPane_mask.png result_MiamiMarcumCenter_${threads}.png true 50 64
    done
done

echo "----------------------[ Checking for memory leak ]----------------"

export OMP_NUM_THREADS=1
echo "------------[ Valgrind TEST(ImageSearch) ]---------------"
valgrind --leak-check=yes ./ImageSearch TestImage.png and_mask.png result_TestImage_1.png true 75 16
echo "------------[ Valgrind TEST(WindowPane_mask.png) ]---------------"
valgrind --leak-check=yes ./ImageSearch MiamiMarcumCenter.png WindowPane_mask.png result_MiamiMarcumCenter_1.png true 50 64

#end of script
