
# 1 8-day mean no considered QC
# check http://www.icess.ucsb.edu/modis/LstUsrGuide/usrguide_8dcmg.html#qa
# and QC at http://www.icess.ucsb.edu/modis/LstUsrGuide/usrguide_1dcmg.html#Table_17

# for DAY  in $(cat  /lustre/scratch/client/fas/sbsc/ga254/dataproces/LST/geo_file/list_day.txt ) ; do qsub -v DAY=$DAY /home/fas/sbsc/ga254/scripts/LST/sc1_wget_MYOD11C2.sh ; done 

#PBS -S /bin/bash
#PBS -q fas_devel
#PBS -l walltime=8:00:00
#PBS -l nodes=1:ppn=8
#PBS -V
#PBS -o /lustre0/scratch/ga254/stdout
#PBS -e /lustre0/scratch/ga254/stderr


# export DAY=$1

export DAY=$DAY

export HDFMOD11C2=/lustre/scratch/client/fas/sbsc/ga254/dataproces/LST/MOD11C2
export HDFMYD11C2=/lustre/scratch/client/fas/sbsc/ga254/dataproces/LST/MYD11C2
export LST=/lustre/scratch/client/fas/sbsc/ga254/dataproces/LST
export RAMDIR=/dev/shm

# seq 2000 2014

seq 2000 2001  | xargs -n 1 -P 8 bash -c $' 
YEAR=$1

wget -N   -P   $RAMDIR   ftp://ladsweb.nascom.nasa.gov/allData/5/MOD11C2/$YEAR/$DAY/*.hdf
wget -N   -P   $RAMDIR  ftp://ladsweb.nascom.nasa.gov/allData/5/MYD11C2/$YEAR/$DAY/*.hdf

file=$(ls $RAMDIR/MOD11C2.A${YEAR}${DAY}*.hdf  )
filename=$(basename $file .hdf)

gdal_translate   -co COMPRESS=LZW -co ZLEVEL=9    HDF4_EOS:EOS_GRID:"$file":MODIS_8DAY_0.05DEG_CMG_LST:LST_Day_CMG  $HDFMOD11C2/$YEAR/$filename.tif 
gdal_translate   -co COMPRESS=LZW -co ZLEVEL=9    HDF4_EOS:EOS_GRID:"$file":MODIS_8DAY_0.05DEG_CMG_LST:QC_Day  $HDFMOD11C2/$YEAR/${filename}_QC.tif 

file=$(ls $RAMDIR/MYD11C2.A${YEAR}${DAY}*.hdf  )
filename=$(basename $file .hdf)

gdal_translate   -co COMPRESS=LZW -co ZLEVEL=9    HDF4_EOS:EOS_GRID:"$file":MODIS_8DAY_0.05DEG_CMG_LST:LST_Day_CMG  $HDFMYD11C2/$YEAR/$filename.tif 
gdal_translate   -co COMPRESS=LZW -co ZLEVEL=9    HDF4_EOS:EOS_GRID:"$file":MODIS_8DAY_0.05DEG_CMG_LST:QC_Day  $HDFMYD11C2/$YEAR/${filename}_QC.tif 

rm $RAMDIR/M?D11C2.A${YEAR}${DAY}*.hdf

' _ 

echo start the pkcomposite /lustre/scratch/client/fas/sbsc/ga254/dataproces/LST/MOYD11C2/LST_Day_CMG_day$DAY.tif 

pkcomposite $(ls $LST/*/*/M?D11C2.A20??${DAY}.005.*[0-9].tif | xargs -n 1 echo -i ) -file observations -ot Float32   -co COMPRESS=LZW -co ZLEVEL=9  -cr mean -dstnodata 0 -srcnodata 0 -o /lustre/scratch/client/fas/sbsc/ga254/dataproces/LST/MOYD11C2/LST_Day_CMG_day$DAY.tif 

