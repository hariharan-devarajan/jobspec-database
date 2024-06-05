#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=stage2

source ~/.bashrc
conda activate amber

m=$1

##ADD CONECT, REMOVE HYD99
mkdir ../stg2_process/${m%.*}_results
rm ../stg2_process/${m%.*}_results/*
cp $m ../stg2_process/${m%.*}_results
cd ../stg2_process/${m%.*}_results
cp ../unclash.py ./
cp ../valency.py ./

sed -i '/END/d' $m
sed -i '/MASTER/d' $m
grep -n -m 2 -e C94 -e O94 -e N94 -e P94 -e S94 $m >> ${m%.*}_in.file
grep -n -m 2 -e C95 -e O95 -e N95 -e P95 -e S95 $m >> ${m%.*}_in2.file
tr -d ' ' < ${m%.*}_in.file | cut -d':' -f1 | sort -u > ${m%.*}_out.file
tr -d ' ' < ${m%.*}_in2.file | cut -d':' -f1 | sort -u > ${m%.*}_out2.file
var1=$(awk 'NR==1' ${m%.*}_out.file);
var2=$(awk 'NR==1' ${m%.*}_out2.file);

var1=$((var1-3));
var2=$((var2-3));

if [ ${#var1} -gt 4 ] && [ ${#var2} -gt 4 ];
then
        string="CONECT${var1}${var2}";
elif [ ${#var1} -gt 4 ] && [ ${#var2} -lt 5 ];
then
        string="CONECT${var1} ${var2}";
elif [ ${#var2} -gt 4 ] && [ ${#var1} -lt 5 ];
then
        string="CONECT ${var1}${var2}";
else
        string="CONECT ${var1} ${var2}";
fi;

echo "$string" >> $m;
echo "END" >> $m;
rm ${m%.*}_in.file ${m%.*}_in2.file ${m%.*}_out.file ${m%.*}_out2.file

python valency.py $m
mv valency_${m} $m
python unclash.py $m
mv unclashed_${m} $m

sed -i '/HETATM/{/UNL/!d}' $m
sed -i "/TER.*CCS\|CCS.*TER/d" $m
sed -i "/TER.*PTR\|PTR.*TER/d" $m
sed -i "/TER.*TPO\|TPO.*TER/d" $m
sed -i "/TER.*DAR\|DAR.*TER/d" $m
sed -i "/TER.*SEP\|SEP.*TER/d" $m
sed -i "/TER.*CSX\|CSX.*TER/d" $m

$SCHRODINGER/utilities/prepwizard $m nhyd99_${m} -NOJOBID -LOCAL -noepik -noccd -f 3 -j STAGE2_${m%.*} -rmsd 5.0 -WAIT

mkdir ../../process_2/nhyd99_${m}_results
mv nhyd99_${m} ../../process_2/nhyd99_${m}_results
rm *
cd ../../process_2//nhyd99_${m}_results

$SCHRODINGER/utilities/prepwizard nhyd99_${m} out_nhyd99_${m} -NOJOBID -LOCAL -f 3 -j stage3_${m%.*} -rmsd 5.0
mv out_nhyd99_${m} ../../process_3
rm *


cd ../../process_3

sbatch maestro4.sh out_nhyd99_${m}
