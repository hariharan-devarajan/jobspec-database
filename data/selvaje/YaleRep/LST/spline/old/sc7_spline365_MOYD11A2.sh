
#  qsub -v SENS=MYD  /u/gamatull/scripts/LST/spline/sc7_spline365_MOYD11A2.sh

#PBS -S /bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=24
#PBS -l walltime=8:00:00
#PBS -V
#PBS -o /nobackup/gamatull/stdout
#PBS -e /nobackup/gamatull/stderr

echo sc2_spline_MOYD11A2 

export  SENS=${SENS}

export INSENS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_splinefill_merge
export OUTSENS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_spline15

export RAMDIR=/dev/shm


# replicate the dataset 3 times    ( in case of using the full data series add the first day at the end of the series ; echo .... )

rm -f /dev/shm/*

# attenzione riattivere solo in caso di rerun e dopo il sc7_maskIndia
# echo 153 161 169 177 185 193 201 209 217 225 | xargs -n 1 -P 10 bash -c $' 
# DAY=$1

# mv $INSENS/LST_${SENS}_akima_day${DAY}_allobs.tif           $INSENS/MaskIndia_nosmooth  
# mv $INSENS/LST_${SENS}_akima_day${DAY}_allobs_MskIndia.tif  $INSENS/LST_${SENS}_akima_day${DAY}_allobs.tif

# ' _ 


awk '{  print $1   }' /nobackupp8/gamatull/dataproces/LST/geo_file/tile_lat_long_10d.txt  | xargs -n 1 -P  6  bash -c $' 

tile=$1

geo_string=$(grep $tile /nobackupp8/gamatull/dataproces/LST/geo_file/tile_lat_long_20d.txt | awk \'{ print $4 , $7 , $6 , $5 }\')

# geo_string=$(echo -161 70 -160 71 )   # insert for validation 

# the nr variable addjust the number of day to put at the befining or at the end of the temporal series. 
# nr change in accordance to the BW. controllare il risultato con le vaidation procedure.

nr=36
nrt=$(expr 47 - $nr)

gdalbuildvrt -te $geo_string   -separate   -overwrite  $RAMDIR/LST_${SENS}_$tile.vrt   $( echo $(awk -v nr=$nr -v  INSENS=$INSENS -v SENS=$SENS   \'{ if (NR>nr) { print INSENS "/LST_" SENS "_akima_day" $1 "_allobs.tif" } }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk  -v  INSENS=$INSENS -v SENS=$SENS \'{ print   INSENS "/LST_" SENS "_akima_day" $1 "_allobs.tif"   }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk -v nrt=$nrt  -v  INSENS=$INSENS -v SENS=$SENS   \'{ if (NR<=nrt)  print  INSENS "/LST_" SENS "_akima_day" $1 "_allobs.tif"   }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ) ) 

BW=30

echo start the spline for $tile  

pkfilter -of GTiff  -ot Float32 -co BIGTIFF=YES  -co COMPRESS=LZW -co ZLEVEL=9 -nodata 0  \
$( echo $(awk -v nr=$nr  \'{ if (NR>nr) {  print "-win" ,  int ($1)} }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk \'{ print "-win" ,  int ($1)+365 }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk -v nrt=$nrt   \'{ if (NR<=nrt)   print "-win" ,  int ($1)+365+365 }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ))\
$( for day in $(seq 366 731 ) ; do  echo -n "-wout $day -fwhm $BW " ; done ) \
-i  $RAMDIR/LST_${SENS}_$tile.vrt   -o $RAMDIR/LST_${SENS}_spline_$tile.tif -interp akima_periodic

# validation procedure .  se si campbia il bw controllareil nr 
# gdal_translate $RAMDIR/LST_${SENS}_$tile.vrt  $RAMDIR/LST_${SENS}_$tile.tif 

# paste <(awk \'{ print  int ($1) }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk \'{ print  int ($1)+365 }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk \'{ print  int ($1)+365+365 }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt  ; echo  1096 )      <(gdallocationinfo   -valonly  $RAMDIR/LST_MYD_h00v00.tif   100 100 )   > full_serie.txt 

# paste <(echo  380 ; echo  410 ; echo 439 ; echo 470 ; echo 500 ; echo 531 ; echo 561 ; echo 592 ; echo 623 ; echo 653 ; echo 684 ; echo 714 ; echo 745) <( gdallocationinfo  -valonly $RAMDIR/LST_MYD_spline_h00v00.tif 100 100   ) > month_serie.txt 

# head -1  month_serie.txt
# tail -1  month_serie.txt

mv  $RAMDIR/LST_${SENS}_spline_$tile.tif    $OUTSENS/LST_${SENS}_spline_$tile.tif 
rm -f  $RAMDIR/LST_${SENS}_spline_$tile.tif   $RAMDIR/LST_${SENS}_$tile.vrt 

' _ 

exit 

rm -f /dev/shm/*

gdalbuildvrt  -overwrite   $RAMDIR/LST_${SENS}_spline.vrt   $OUTSENS/LST_${SENS}_spline_*.tif 

seq 1 13  | xargs -n 1 -P  13 bash -c $'  

band=$1 

echo start the merging action for month $band

gdal_translate -b $band  -co COMPRESS=LZW -co ZLEVEL=9   $RAMDIR/LST_${SENS}_spline.vrt  /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_spline15_merge/LST_${SENS}_spline_month$band.tif 
pksetmask -of GTiff -co COMPRESS=LZW -co ZLEVEL=9 \
-m /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk/${SENS}_LST3k_count_yessea_ct.tif                     -msknodata 255   -nodata -1 \
-m /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk/${SENS}_LST3k_count_yessea_ct.tif                     -msknodata 0     -nodata -1 \
-m /nobackup/gamatull/dataproces/LST/${SENS}11A2_mean_msk/${SENS}_LST3k_mask_daySUM_wgs84_allobs_mask5noobs.tif   -msknodata 1     -nodata -1 \
-i /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_spline15_merge/LST_${SENS}_spline_month$band.tif   -o /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_spline15_merge/LST_${SENS}_spline_month${band}_tmp.tif 

mv /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_spline15_merge/LST_${SENS}_spline_month${band}_tmp.tif   /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_spline15_merge/LST_${SENS}_spline_month${band}.tif 

' _

rm -f $RAMDIR/LST_${SENS}_spline.vrt  /dev/shm/*


