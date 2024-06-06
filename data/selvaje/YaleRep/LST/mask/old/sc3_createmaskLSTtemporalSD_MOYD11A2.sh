# for Nr in $(seq 27 27 ) ; do qsub -v Nr=$Nr,SENS=MOD /u/gamatull/scripts/LST/mask/sc3_createmaskLSTtemporalSD_MOYD11A2.sh    ; done
# 
# create a mask that is able to identify pixel having temporal sd > than 3 times the sd 

# temp_wind=3 ; (tail -$temp_wind  /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; cat /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; head -$temp_wind /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt) > /nobackup/gamatull/dataproces/LST/geo_file/list_day_replicate.txt

#PBS -S /bin/bash
#PBS -q normal 
#PBS -l select=1:ncpus=23
#PBS -l walltime=8:00:00
#PBS -V
#PBS -o   /nobackup/gamatull/stdout
#PBS -e   /nobackup/gamatull/stderr

export SENS=MYD
export IN=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean
export OUT=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_sd_msk

export RAMDIR=/dev/shm
export Nr=$Nr
rm -f  /dev/shm/*

echo start the pkcomposite for $DAY $SENS

export temp_wind=3
export start_wind=$(expr $Nr )
export end_wind=$(expr $temp_wind + $Nr + $temp_wind )

echo $start_wind  $end_wind

DAY=$( awk -v Nr=$Nr  '{ if (NR==Nr)  print $1  }'  /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt )

cat    /nobackupp8/gamatull/dataproces/LST/geo_file/tile_lat_long_20d_land.txt    | xargs -n 1 -P  23  bash -c $' 

# india tile ul=h24v06  ll=h24v08  ur=h26v06 lr=h26v08 
tile=$1
geo_string=$(grep $tile /nobackupp8/gamatull/dataproces/LST/geo_file/tile_lat_long_20d.txt | awk \'{ print $4 , $7 , $6 , $5 }\')

gdalbuildvrt -separate  -te $geo_string  -overwrite  $RAMDIR/LST_${SENS}_$tile.vrt   $( awk  -v start_wind=$start_wind -v end_wind=$end_wind  \'{ if ( NR>=start_wind  &&  NR<=end_wind) print $1   }\'     /nobackup/gamatull/dataproces/LST/geo_file/list_day_replicate.txt   | xargs  -n 1 bash -c $\'  echo ${IN}/LST_MYD_QC_day${1}_wgs84.tif \' _ )

pkcomposite  -of GTiff \
$(awk  -v start_wind=$start_wind -v end_wind=$end_wind  \'{ if ( NR>=start_wind  &&  NR<=end_wind) print $1   }\'     /nobackup/gamatull/dataproces/LST/geo_file/list_day_replicate.txt   | xargs  -n 1 bash -c $\'  echo -i ${IN}/LST_MYD_QC_day${1}_wgs84.tif \' _ ) \
-file 1 -ot Float32  -co  COMPRESS=LZW -co ZLEVEL=9  -cr mean  -dstnodata 0 -bndnodata 0 -srcnodata 0 -o $RAMDIR/LST_${SENS}_mn_day${DAY}_tile${tile}.tif

pkcomposite  -of GTiff \
$(awk  -v start_wind=$start_wind -v end_wind=$end_wind  \'{ if ( NR>=start_wind  &&  NR<=end_wind) print $1   }\'     /nobackup/gamatull/dataproces/LST/geo_file/list_day_replicate.txt   | xargs  -n 1 bash -c $\'  echo -i ${IN}/LST_MYD_QC_day${1}_wgs84.tif \' _ ) \
-file 1 -ot Float32  -co  COMPRESS=LZW -co ZLEVEL=9  -cr stdev  -dstnodata 0 -bndnodata 0 -srcnodata 0 -o $RAMDIR/LST_${SENS}_sd_day${DAY}_tile${tile}.tif

' _ 

gdalbuildvrt    -overwrite   $RAMDIR/LST_${SENS}_sd_day${DAY}.vrt    $RAMDIR/LST_${SENS}_sd_day${DAY}_tile?.tif 
gdal_translate  -co COMPRESS=LZW -co ZLEVEL=9   $RAMDIR/LST_${SENS}_sd_day${DAY}.vrt  $OUT/LST_${SENS}_sd_day${DAY}.tif 

gdalbuildvrt    -overwrite   $RAMDIR/LST_${SENS}_mn_day${DAY}.vrt    $RAMDIR/LST_${SENS}_mn_day${DAY}_tile?.tif
gdal_translate  -co COMPRESS=LZW -co ZLEVEL=9   $RAMDIR/LST_${SENS}_mn_day${DAY}.vrt  $OUT/LST_${SENS}_mn_day${DAY}.tif

rm -f  /dev/shm/*