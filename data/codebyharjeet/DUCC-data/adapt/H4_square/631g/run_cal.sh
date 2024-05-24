#!/bin/bash

#SBATCH --nodes=1
#SBATCH -p normal_q
#SBATCH --cpus-per-task=16
#SBATCH -t 20:00:00
##SBATCH --account=personal
#SBATCH --account=nmayhall_group
#SBATCH --job-name test13
#SBATCH --exclusive

# export NTHREAD=16


# allow time to load modules, this takes time sometimes for some reason
#sleep 10

hostname

module reset; module load intel/2019b
#module reset
#module load site/tinkercliffs-rome/easybuild/setup     #only for tinkercliffs
#module load site/tinkercliffs/easybuild/setup  #only for tinkercliffs
#module load Anaconda3/2020.11
#module load gcc/9.2.0

source activate myenv

echo "Usage: sbatch submit.sh {input file} {data file} {data file}"

# set these to 1 if you are using julia threads heavily to avoid oversubscription
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

export INFILE=$1
export OUTFILE="${INFILE}.out"
export WORKDIR=$(pwd)

echo $INFILE
echo $OUTFILE
echo $WORKDIR
echo $TMPDIR

cp $INFILE $TMPDIR/
if [ "$2" ]
then
        cp $2 $TMPDIR/
fi
if [ "$3" ]
then
        cp $3 $TMPDIR/
fi
cd $TMPDIR


#Start an rsync command which runs in the background and keeps a local version of the output file up to date
touch $OUTFILE
while true; do rsync -av $OUTFILE $WORKDIR/"${INFILE}.${SLURM_JOB_ID}.out"; sleep 60; done &


#julia --project=$JULIAENV -t $NTHREAD $INFILE >& $OUTFILE
python $INFILE >& $OUTFILE

cp $OUTFILE $WORKDIR/"${INFILE}.out"
rm $WORKDIR/"${INFILE}.${SLURM_JOB_ID}.out"

mkdir $WORKDIR/"${INFILE}.${SLURM_JOB_ID}.scr"

cp -r * $WORKDIR/"${INFILE}.${SLURM_JOB_ID}.scr"
rm -r *

#moving standard output slurm file to specific job directory
mv $WORKDIR/"slurm-${SLURM_JOB_ID}.out" $WORKDIR/"${INFILE}.${SLURM_JOB_ID}.scr"

exit


