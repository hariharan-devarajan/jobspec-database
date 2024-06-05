#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=stage1

i=$1
#o=$2
#for i in amb_*xx${u}.pdb;do

mkdir ../temp_process/${i%.*}_stg1res
rm ../temp_process/${i%.*}_stg1res/*
mv $i ../temp_process/${i%.*}_stg1res
cd ../temp_process/${i%.*}_stg1res

$SCHRODINGER/utilities/prepwizard -WAIT -NOJOBID -noepik -noimpref -rmsd 5.0 -f 3 -j stage1_${i%.*} ${i%.*}.pdb mae_${i%.*}.pdb

mv mae_${i%.*}.pdb ../../process/
mv ${i%.*}.pdb ../../process/
rm *

cd ../../process/ 

sbatch maestro3.sh mae_${i%.*}.pdb
