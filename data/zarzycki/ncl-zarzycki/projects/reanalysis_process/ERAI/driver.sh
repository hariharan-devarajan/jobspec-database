#!/bin/bash

##=======================================================================
#BSUB -a poe                     # use LSF openmp elim
#BSUB -N
#BSUB -n 1                      # yellowstone setting
#BSUB -o out.%J                  # output filename
#BSUB -e out.%J                  # error filename
#BSUB -q geyser                 # queue
#BSUB -J sub_ncl 
#BSUB -W 23:58                    # wall clock limit
#BSUB -P P54048000               # account number

################################################################

ERAIBASEDIR=/glade/collections/rda/data//ds627.0/
SYMDIR=/glade/u/home/zarzycki/scratch/ERAIsym/
OUTBASE=/glade/scratch/zarzycki/h1files/ERAI/
YYYY=${1}

### Generate time array file
arrayFileName=timesArray_${YYYY}.txt
rm ${arrayFileName}
start=$(date -u --date '1 jan '${YYYY}' 0:00' +%s)
stop=$(date -u --date '31 dec '${YYYY}' 18:00' +%s)
for t in $(seq ${start} 21600 ${stop})
do
  thisDate=`date -u --date @${t} +'%Y%m%d%H'`
  echo $thisDate >> ${arrayFileName}
done

### Symlink ERAI files
mkdir -p ${SYMDIR}/${YYYY}
rm ${SYMDIR}/${YYYY}/*.grb2

declare -a anl_mdl_arr=("ml" "pl" "sfc")
for i in "${anl_mdl_arr[@]}"
do
  FILES=${ERAIBASEDIR}/ei.oper.an.${i}/${YYYY}*/ei.oper.an.*regn128sc*
  for f in $FILES
  do
    echo "Processing $f file..."
    a=$(basename $f)
    ln -s ${f} ${SYMDIR}/${YYYY}/${a}.grb2
  done
done

### Make output dir
mkdir -p ${OUTBASE}/${YYYY}

# doo NCL script
while read p; do
  f=${OUTBASE}/${YYYY}/ERAI.h1.${p}.nc
  ncl generateTrackerFilesERAI-lite.ncl 'dateListArr="'${p}'"' 'SYMDIR="'${SYMDIR}'/'${YYYY}'"' 'OUTDIR="'${OUTBASE}'/'${YYYY}'"' 
done < ${arrayFileName}

rm -rf ${SYMDIR}/${YYYY}


CONFIG=ERAI
OUTFILEDIR=${OUTBASE}/${YYYY}/

### Generate time array file
arrayFileName=test_timesArray_${YYYY}.txt
rm ${arrayFileName}
start=$(date -u --date '1 jan '${YYYY}' 0:00' +%s)
stop=$(date -u --date '31 dec '${YYYY}' 0:00' +%s)

for t in $(seq ${start} 86400 ${stop})
do
  thisDate=`date -u --date @${t} +'%Y%m%d'`
  echo $thisDate >> ${arrayFileName}
done

mkdir ${OUTFILEDIR}/TMP
mv ${OUTFILEDIR}/*.nc ${OUTFILEDIR}/TMP

while read p; do
  echo $p
  ncrcat ${OUTFILEDIR}/TMP/${CONFIG}.h1.${p}*.nc ${OUTFILEDIR}/${CONFIG}.h1.${p}.nc
done < ${arrayFileName}

rm ${arrayFileName}
rm -rf ${OUTFILEDIR}/TMP/

