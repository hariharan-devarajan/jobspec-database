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

OUTBASE=/glade/scratch/zarzycki/h1files/CR20/
YYYY=${1}

CONFIG=CR20
OUTFILEDIR=/glade/scratch/zarzycki/h1files/CR20/${YYYY}/
### Generate time array file
arrayFileName=test_timesArray_${YYYY}.txt
rm ${arrayFileName}
start=$(date -u --date '1 jan '${YYYY}' 0:00' +%s)
stop=$(date -u --date '31 dec '${YYYY}' 18:00' +%s)
for t in $(seq ${start} 21600 ${stop})
do
  thisDate=`date -u --date @${t} +'%Y%m%d%H'`
  echo $thisDate >> ${arrayFileName}
done

mkdir -p ${OUTFILEDIR}
ncl generateTrackerFiles20CRV3-lite.ncl 'YEAR="'${YYYY}'"'

######################
dailyarrayFileName=daily_timesArray_${YYYY}.txt
start=$(date -u --date '1 jan '${YYYY}' 0:00' +%s)
stop=$(date -u --date '31 dec '${YYYY}' 0:00' +%s)
for t in $(seq ${start} 86400 ${stop})
do
  thisDate=`date -u --date @${t} +'%Y%m%d'`
  echo $thisDate >> ${dailyarrayFileName}
done

mkdir ${OUTFILEDIR}/TMP
mv -v ${OUTFILEDIR}/*.nc ${OUTFILEDIR}/TMP

while read p; do
  echo $p
  ncrcat -O ${OUTFILEDIR}/TMP/${CONFIG}.h1.${p}*.nc ${OUTFILEDIR}/${CONFIG}.h1.${p}.nc
done < ${dailyarrayFileName}

rm ${arrayFileName}
rm ${dailyarrayFileName}
rm -rf ${OUTFILEDIR}/TMP/
