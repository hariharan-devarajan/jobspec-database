#!/usr/bin/env bash
#SBATCH --partition=long											### Partition (like a queue in PBS)
#SBATCH --job-name=demultiplexGraphs								### Job Name
#SBATCH --output=/projects/bgmp/maddyg/demultiplex/files/graphs.out	### File in which to stor job output
#SBATCH --error=/projects/bgmp/maddyg/demultiplex/files/graphs.err	### Filein which to store job error messages
#SBATCH --time=4-00:00:00											### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1													### Number of nodes needed for the job
#SBATCH --ntasks-per-node=14										### Number of tasks to be launched per Node
#SBATCH --mail-user=maddyg@uoregon.edu 								### Email to recieve notifications
#SBATCH --mail-type=BEGIN,END,FAIL									### When should emails be sent

module purge
ml easybuild intel/2017a Python/3.6.1

fType1=read
fType2=index

file1=/projects/bgmp/shared/2017_sequencing/1294_S1_L008_R1_001.fastq.gz
name1=R1

file2=/projects/bgmp/shared/2017_sequencing/1294_S1_L008_R2_001.fastq.gz
name2=R2

file3=/projects/bgmp/shared/2017_sequencing/1294_S1_L008_R3_001.fastq.gz
name3=R3

file4=/projects/bgmp/shared/2017_sequencing/1294_S1_L008_R4_001.fastq.gz
name4=R4

python3 ./../python/demultiplex.py -f $file1 -t $fType1 -n $name1
python3 ./../python/demultiplex.py -f $file2 -t $fType2 -n $name2
python3 ./../python/demultiplex.py -f $file3 -t $fType2 -n $name3
python3 ./../python/demultiplex.py -f $file4 -t $fType1 -n $name4