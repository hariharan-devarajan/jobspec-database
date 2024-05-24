#!/bin/bash

#SBATCH --job-name=YAMP
#SBATCH --mail-user=lptolik@gmail.com
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=YAMP_%j.out
#SBATCH --error=YAMP_%j.err

if test "$#" -ne 4; then
    echo "Script requires 4 parameters:"
    echo "Read1, Read2, sample name and output directory"
    echo "In that exact order"
    exit 2
fi

cwdir=`pwd`

R1=$1
R2=$2
pref=$3
odir=$4

echo "Reads:"
echo "$R1"
echo "$R2"
echo "Output directory:"
echo "$odir"
echo "Sample name: $pref"

wd=$PWD
# create a temporary directory for this job and save the name 
tempdir=$(mktemp -d /flash/GoryaninU/lptolik/YAMP.XXXXXX)
echo $tempdir

# enter the temporary directory
cd $tempdir


cp "$wd/nextflow.group.config" "$wd/YAMP.nf" "./"
cp -r "$wd/bin" "./tmp$pref/"
mv nextflow.group.config nextflow.config
SING_WD=./work/singularity
SING_IMG=$SING_WD/alesssia-yampdocker.img
if [ -f "$SING_IMG" ]; then
    echo "$SING_IMG exist"
else 
    echo "$SING_IMG does not exist"
    mkdir -p $SING_WD
    cp /bucket/GoryaninU/Software/YAMP/alesssia-yampdocker.img $SING_IMG
fi

module load singularity

mkdir -p "./$pref/"

/bucket/GoryaninU/Software/nextflow run YAMP.nf --reads1 $R1 --reads2 $R2 --prefix $pref --outdir "./$pref" --mode complete -with-singularity docker://alesssia/yampdocker

mkdir -p "./tmp$pref/"

datdir=`echo $wd| sed s/flash/bucket/`

ssh deigo "mkdir -p $odir/$pref"

# copy our result back to bucket. We need to use "scp"
# to copy the data back as bucket isn't writable directly. 
scp "./$pref/*" deigo:"$odir/$pref"

cd $wd
# Clean up by removing our temporary directory 
rm -r $tempdir
