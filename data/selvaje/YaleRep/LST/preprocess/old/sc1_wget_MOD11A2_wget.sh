# 1 8-day mean no considered QC
# check http://www.icess.ucsb.edu/modis/LstUsrGuide/usrguide_8dcmg.html#qa
# and QC at http://www.icess.ucsb.edu/modis/LstUsrGuide/usrguide_1dcmg.html#Table_17

# for YEAR in $(awk '{ print $2 }'  /nobackupp8/gamatull/dataproces/LST/geo_file/list_day.txt) ; do qsub -v YEAR=$YEAR    -v DAY=$DAY /u/gamatull/scripts/LST/sc1_wget_MYD11A2.sh ; done 

# infor at http://www.nas.nasa.gov/hecc/support/kb/ 

#  -q   normal, debug, long, and low.   devel 2hours 

#PBS -S /bin/bash
#PBS -q devel
#PBS -l select=1:ncpus=15
#PBS -l walltime=2:00:00
#PBS -V
#PBS -o   /nobackup/gamatull/stdout
#PBS -e   /nobackup/gamatull/stderr

echo "DAY  $DAY /u/gamatull/scripts/LST/sc1_wget_MYOD11A2.sh " >  /nobackup/gamatull/stdnode/job_start_$PBS_JOBID.txt
# checkjob -v $PBS_JOBID                                          >> /nobackup/gamatull/stdnode/job_start_$PBS_JOBID.txt

export DAY=$2

echo processing dir  ${DAY} 

export HDFMOD11A2=/nobackupp6/aguzman4/climateLayers/MOD11A2.005/MOD11A2
export HDFMYD11A2=/nobackupp6/aguzman4/climateLayers/MYD11A2.005/MYD11A2
export LST=/nobackupp6/aguzman4/climateLayers/MOD11A2.005
export INDIR=/nobackupp6/aguzman4/climateLayers/MOD11A2.005

export RAMDIR=/dev/shm

rm -f /dev/shm/*.hdf   /dev/shm/*.tif  /dev/shm/.listing

export YEAR=$1

echo $YEAR  | xargs -n 1 -P 15  bash -c $' 

YEAR=$1 
SENS=MOD

echo start the download of ftp://ladsweb.nascom.nasa.gov/allData/5/${SENS}11A2/${YEAR}/$DAY/hdf

wget -w 5 --waitretry=4 --random-wait   --no-remove-listing   -o $LST/${SENS}11A2/$YEAR/${SENS}11A2.A${YEAR}${DAY}_hdf.log  -c   -N   -P   $RAMDIR   ftp://ladsweb.nascom.nasa.gov/allData/5/${SENS}11A2/${YEAR}/$DAY/*.hdf  

ls    $RAMDIR/${SENS}11A2.A$YEAR${DAY}.??????.???.*.hdf    >  $LST/${SENS}11A2/$YEAR/${SENS}11A2.A${YEAR}${DAY}_hdf.txt

if [ -s  $LST/${SENS}11A2/$YEAR/${SENS}11A2.A${YEAR}${DAY}_hdf.txt  ] ; then 

for file in   $RAMDIR/${SENS}11A2.A$YEAR${DAY}.??????.???.*.hdf ; do 

filename=$(basename $file .hdf)

gdal_translate -ot  UInt16  -co COMPRESS=LZW -co ZLEVEL=9     HDF4_EOS:EOS_GRID:"$file":MODIS_Grid_8Day_1km_LST:LST_Day_1km   $RAMDIR/${filename}_LST.tif &> /dev/null
gdal_translate -ot  UInt16  -co COMPRESS=LZW -co ZLEVEL=9     HDF4_EOS:EOS_GRID:"$file":MODIS_Grid_8Day_1km_LST:QC_Day        $RAMDIR/${filename}_QC.tif  &> /dev/null

rm -f $file

done 

echo start to spatial merge $YEAR  $DAY 

gdalbuildvrt   -overwrite  $RAMDIR/${SENS}11A2.A${YEAR}${DAY}_LST.vrt    $RAMDIR/${SENS}11A2.A$YEAR${DAY}.??????.???.*_LST.tif  
gdalbuildvrt   -overwrite  $RAMDIR/${SENS}11A2.A${YEAR}${DAY}_QC.vrt     $RAMDIR/${SENS}11A2.A$YEAR${DAY}.??????.???.*_QC.tif  

echo merge the LST and QC

gdalbuildvrt   -overwrite -separate      $RAMDIR/${SENS}11A2.A${YEAR}${DAY}.vrt    $RAMDIR/${SENS}11A2.A${YEAR}${DAY}_LST.vrt    $RAMDIR/${SENS}11A2.A${YEAR}${DAY}_QC.vrt

gdal_translate  -ot  UInt16    -co COMPRESS=DEFLATE -co ZLEVEL=9   -co  PREDICTOR=2  $RAMDIR/${SENS}11A2.A${YEAR}${DAY}.vrt  $LST/${SENS}11A2/$YEAR/${SENS}11A2.A${YEAR}${DAY}.tif

# rm -f  $RAMDIR/*11A2.A*.??????.???.*.tif  $RAMDIR/*.vrt   $RAMDIR/${SENS}11A2.A$YEAR${DAY}.??????.???.*.hdf 

fi

' _ 

echo "MMDD $MMDD /u/gamatull/scripts/LST/sc1_wget_MYOD11A2.sh " >  /nobackup/gamatull/stdnode/job_end_$PBS_JOBID.txt
# checkjob -v $PBS_JOBID >>  /nobackup/gamatull/stdnode/job_end_$PBS_JOBID.txt

exit 
