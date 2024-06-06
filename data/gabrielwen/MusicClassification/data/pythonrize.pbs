#!/bin/bash

#PBS -V
#PBS -S /bin/bash
#PBS -N mussic
#PBS -l nodes=1:ppn=2
#PBS -l walltime=04:00:00
#PBS -l mem=2GB
#PBS -q s48
#PBS -M yss265@nyu.edu
#PBS -m bea
#PBS -e localhost:${PBS_O_WORKDIR}/${PBS_JOBNAME}.e${PBS_JOBID}
#PBS -o localhost:${PBS_O_WORKDIR}/${PBS_JOBNAME}.o${PBS_JOBID}


cd /home/yss265/MusicClassification/data
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -zxvf genres.tar.gz

cd /home/yss265/MusicClassification/data/genres
module load sox/intel/14.4.1
cp ../process.sh .
sh process.sh

cd /home/yss265/MusicClassification/data
module load scipy/intel/0.13.3
python pythonrize.py

cd /home/yss265/MusicClassification/data
rm -rf genres genres_processed genres.tar.gz

exit 0;
