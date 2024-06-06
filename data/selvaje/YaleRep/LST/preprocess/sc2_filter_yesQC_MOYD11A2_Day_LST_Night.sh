# for DAY in $(cat /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt  ) ; do qsub -v DAY=$DAY,SENS=MOD,DN=$DN /u/gamatull/scripts/LST/preprocess/sc2_filter_yesQC_MOYD11A2_Day_LST_Night.sh ; done

#PBS -S /bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=16
#PBS -l walltime=6:00:00
#PBS -V
#PBS -o   /nobackup/gamatull/stdout
#PBS -e   /nobackup/gamatull/stderr

export DAY=$1
export SENS=$2
export DN=$3   # Day Nig
export INSENS=/nobackupp6/aguzman4/climateLayers/${SENS}11A2.005/${SENS}11A2
export OUTSENS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean

export RAMDIR=/dev/shm
rm -f $RAMDIR/*.tif  $RAMDIR/*.vrt

echo start the pkcomposite for $DAY $SENS

echo 0         0 5400 10800 a  >  $RAMDIR/tiles_xoff_yoff.txt
echo 5400      0 5400 10800 b  >> $RAMDIR/tiles_xoff_yoff.txt
echo 10800     0 5400 10800 c  >> $RAMDIR/tiles_xoff_yoff.txt
echo 16200     0 5400 10800 d  >> $RAMDIR/tiles_xoff_yoff.txt
echo 21600     0 5400 10800 e  >> $RAMDIR/tiles_xoff_yoff.txt
echo 30000  7000 2000 1000 f  >> $RAMDIR/tiles_xoff_yoff.txt  # 27000
echo 32400     0 5400 10800 g  >> $RAMDIR/tiles_xoff_yoff.txt
echo 37800     0 5400 10800 h  >> $RAMDIR/tiles_xoff_yoff.txt
echo 0      10800 5400 10800 i  >> $RAMDIR/tiles_xoff_yoff.txt
echo 5400   10800 5400 10800 l  >> $RAMDIR/tiles_xoff_yoff.txt
echo 10800  10800 5400 10800 m  >> $RAMDIR/tiles_xoff_yoff.txt
echo 16200  10800 5400 10800 n  >> $RAMDIR/tiles_xoff_yoff.txt
echo 21600  10800 5400 10800 o  >> $RAMDIR/tiles_xoff_yoff.txt
echo 27000  10800 5400 10800 p  >> $RAMDIR/tiles_xoff_yoff.txt
echo 32400  10800 5400 10800 q  >> $RAMDIR/tiles_xoff_yoff.txt
echo 37800  10800 5400 10800 r  >> $RAMDIR/tiles_xoff_yoff.txt

cat $RAMDIR/tiles_xoff_yoff.txt | head -6 | tail -1   | xargs -n 5 -P 16 bash -c $' 

xoff=$1
yoff=$2
xsize=$3
ysize=$4
tile=$5

for file in $INSENS/*/${SENS}11A2.A*_$DN.tif ; do 

filename=$(basename $file .tif)
gdal_translate    -co  COMPRESS=LZW -co ZLEVEL=9 -co  INTERLEAVE=BAND   -srcwin  $xoff $yoff $xsize $ysize   $file   $RAMDIR/${filename}_tile${tile}_$DN.tif
done 

echo start the pcomposite sens ${SENS} day ${DAY} tile ${tile}

# -b not usede if not the b2 can not be used   see fir qc /nobackupp8/gamatull/dataproces/LST/MYD11A2/QC_decimal_bynary.txt 
#  in this case  -bndnodata 1 e la 2 banda

rm -f $RAMDIR/LST_${SENS}_QC_day${DAY}_$DN.vrt
gdalbuildvrt  -overwrite -separate  -b 1  $RAMDIR/LST_${SENS}_QC_dayALL_tile${tile}_$DN.vrt  $RAMDIR/${SENS}11A2.A???????_Day_tile${tile}_$Day.tif    2> /dev/null 

# ~/bin/pkfilter  -co  COMPRESS=LZW -co ZLEVEL=9 -co  INTERLEAVE=BAND    --filter savgolay -nl 2  -nr 2 -ld 0 -m 2 -i -dx 1 -dy 1 -dz 12 -i $RAMDIR/LST_${SENS}_QC_day${DAY}_$DN.vrt -o $RAMDIR/LST_${SENS}_QC_day${DAY}_${DN}_fsg.tif

# pkcomposite $(ls $RAMDIR/${SENS}11A2.A*${DAY}_tile${tile}_$DN.tif | xargs -n 1 echo -i )  -file 1  -ot Float32   -co  COMPRESS=LZW -co ZLEVEL=9  -cr mean -dstnodata 0  -bndnodata 1 -srcnodata 2 -srcnodata 3  -srcnodata  193 -srcnodata 209 -srcnodata 225       -o $RAMDIR/LST_${SENS}_QC_day${DAY}_tile${tile}_$DN.tif 

' _ 

exit 


gdalbuildvrt  -overwrite -separate  -b 1  $RAMDIR/LST_${SENS}_QC_dayALL_tile${tile}_$DN.vrt  $RAMDIR/${SENS}11A2.A???????_Day_tile${tile}_Day.tif    2> /dev/null

~/bin/pkfilter -f savgolay   -nl 2  -nr 2 -ld 0 -m 2    -of  GTiff  -dx 1 -dy 1 -dz 12   -i $RAMDIR/LST_${SENS}_QC_dayALL_tilef_Day.vrt   -o test.tif
gdallocationinfo -valonly  $RAMDIR/LST_${SENS}_QC_dayALL_tilef_Day.vrt 650 430 > test1.txt
gdallocationinfo -valonly  test.tif 650 430  > test2.txt

gdallocationinfo -valonly  $RAMDIR/LST_${SENS}_QC_dayALL_tilef_Day.vrt 958 796 > nofilter.txt
gdallocationinfo -valonly  test.tif 960 749  > filter.txt 
for file in    /dev/shm/MOD11A2.A???????_Day_tilef_Day.tif  ; do echo ${file:18:7} ;done   > date.txt

paste nofilter.txt date.txt > nofilter_date.txt
paste filter.txt date.txt > filter_date.txt



exit 


rm -f $RAMDIR/tiles_xoff_yoff.txt

gdalbuildvrt  -overwrite  -b 1 -b 3      $RAMDIR/LST_${SENS}_QC_day${DAY}_$DN.vrt      $RAMDIR/LST_${SENS}_QC_day${DAY}_tile?_$DN.tif
gdal_translate  -srcwin 0 0 43200 21600     -co  COMPRESS=LZW -co ZLEVEL=9  $RAMDIR/LST_${SENS}_QC_day${DAY}_$DN.vrt    $OUTSENS/LST_${SENS}_QC_day${DAY}_$DN.tif 
rm -f $RAMDIR/LST_${SENS}_QC_day${DAY}_tile?.tif

echo warp the lst  yes qc

gdalbuildvrt  -overwrite  -b 1  $OUTSENS/LST_${SENS}_QC_day${DAY}_$DN.vrt   $OUTSENS/LST_${SENS}_QC_day${DAY}_$DN.tif  

gdalwarp  -srcnodata 0 -dstnodata 0   -t_srs EPSG:4326  -r near  -te -180 -90 +180 +90  -tr 0.008333333333333 0.008333333333333   -multi  -overwrite -co BIGTIFF=YES  -co  COMPRESS=LZW -co ZLEVEL=9    $OUTSENS/LST_${SENS}_QC_day${DAY}_$DN.vrt     $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84t_$DN.tif 
rm -f $OUTSENS/LST_${SENS}_QC_day$DAY.vrt  

gdalwarp  -srcnodata 0 -dstnodata 0  -t_srs EPSG:4326  -r average  -te -180 -90 +180 +90 -tr 0.08333333333333 0.08333333333333   -multi  -overwrite -co BIGTIFF=YES  -co  COMPRESS=LZW -co ZLEVEL=9   $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84t_$DN.tif    $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84k10t_$DN.tif

gdal_translate -co  COMPRESS=LZW -co ZLEVEL=9    $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84t_$DN.tif     $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84_$DN.tif
gdal_translate -co  COMPRESS=LZW -co ZLEVEL=9    $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84k10t_$DN.tif  $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84k10_$DN.tif
rm -f $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84t_$DN.tif   $OUTSENS/LST_${SENS}_QC_day${DAY}_wgs84k10t_$DN.tif 

echo warp the obs yes  qc

gdalbuildvrt  -overwrite  -b 2  $OUTSENS/OBS_${SENS}_QC_day${DAY}_$DN.vrt   $OUTSENS/LST_${SENS}_QC_day${DAY}_$DN.tif  
gdalwarp  -srcnodata 0 -dstnodata 0   -t_srs EPSG:4326  -r near  -te -180 -90 +180 +90   -tr 0.008333333333333 0.008333333333333   -multi  -overwrite -co BIGTIFF=YES  -co  COMPRESS=LZW -co ZLEVEL=9    $OUTSENS/OBS_${SENS}_QC_day${DAY}_$DN.vrt   $RAMDIR/OBS_${SENS}_QC_day${DAY}_wgs84_$DN.tif

rm -f $OUTSENS/OBS_${SENS}_QC_day${DAY}_$DN.vrt

gdal_translate -ot Byte   -co  COMPRESS=LZW -co ZLEVEL=9    $RAMDIR/OBS_${SENS}_QC_day${DAY}_wgs84_$DN.tif  $OUTSENS/OBS_${SENS}_QC_day${DAY}_wgs84_$DN.tif

rm -f  $RAMDIR/*.tif $OUTSENS/LST_${SENS}_QC_day${DAY}_$DN.vrt    $OUTSENS/OBS_${SENS}_QC_day${DAY}_$DN.vrt   
