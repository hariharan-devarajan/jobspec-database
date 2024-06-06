#!/bin/bash
#PBS -q viper
#PBS -N mjzBTRefGenes
#PBS -l walltime=0:15:00
#PBS -l nodes=20
. wgaconfig.conf
# hack near the bottom, selecting only the first 0:9 characters of the file basename
# hack change salmonella dir declaration twice
mkdir $bamDir
mkdir $scripts1Dir
i=0
while read line
do
fastaFiles1[ $i ]="$line"
(( i++ ))
done < <(find ${inputDir}*1.fastq.gz -type f -exec basename {} \;)
i=0
while read line
do
fastaFiles2[ $i ]="$line"
(( i++ ))
done < <(find ${inputDir}*2.fastq.gz -type f -exec basename {} \;)
#ls -m1 salmonella)

arrayLength=${#fastaFiles1[@]}
echo $arrayLength " fastaFiles1"


# echo paired end reads
# add a check later
for (( k=0; k < $arrayLength; k++))
do
    echo $k
    echo ${fastaFiles1[k]:0:9}  " " ${fastaFiles2[k]:0:9}
done



for ((i=0; i < $arrayLength; i++))
do
cat > ${scripts1Dir}pbsscript_$i << EOF
#!/bin/bash
#PBS -q viper
#PBS -N mjzBowTie_${i}
#PBS -l walltime=5:01:00
#PBS -l nodes=1
#module load samtools
${bowtie2Dir}bowtie2 -x ${inputRefDir} -p 1 \
-1 $inputDir${fastaFiles1[i]} \
-2 $inputDir${fastaFiles2[i]} \
| ${samtoolsDir}samtools view -bS - > ${bamDir}${fastaFiles1[i]:0:9}.bam
EOF
done

#&
#output_NoExtension=${fastaFiles1[i]:0:9}
#printf -v $outputNoExtension

