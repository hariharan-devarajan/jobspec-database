#  cd /tmp/ ; for SENS in MYD MOD ; do for DN in Day Nig ; do  qsub  -v SENS=$SENS,DN=$DN  /u/gamatull/scripts/LST/spline/sc9_spline15_MOYD11A2.sh  ; done ; done ; cd -
#  qsub -v SENS=MYD,DN=$DN  /u/gamatull/scripts/LST/spline/sc9_spline15_MOYD11A2.sh

#PBS -S /bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=24:model=has
#PBS -l walltime=8:00:00
#PBS -V
#PBS -o /nobackup/gamatull/stdout
#PBS -e /nobackup/gamatull/stderr
#PBS -N sc9_spline15_MOYD11A2.sh

echo /u/gamatull/scripts/LST/spline/sc9_spline15_MOYD11A2.sh

export SENS=${SENS}
export DN=${DN}
export INSENS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_splinefill_merge
export OUTSENS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_spline15
export MSK=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk
export RAMDIR=/dev/shm

# replicate the dataset 3 times    ( in case of using the full data series add the first day at the end of the series ; echo .... )

cleanram 

awk '{ print $1 }' /nobackupp8/gamatull/dataproces/LST/geo_file/tile_lat_long_20d_land.txt  | xargs -n 1 -P  24  bash -c $' 

tile=$1

geo_string=$(grep $tile /nobackupp8/gamatull/dataproces/LST/geo_file/tile_lat_long_20d.txt | awk \'{ print $4 , $7 , $6 , $5 }\')

# geo_string=$(echo -161 70 -160 71 )   # insert for validation 

# the nr variable addjust the number of day to put at the befining or at the end of the temporal series. 
# nr change in accordance to the BW. controllare il risultato con le vaidation procedure.

nr=36
nrt=$(expr 47 - $nr)

# 153 161 169 177 185 193 201 209 217 225 list day that use the _allobs_MskIndia.tif 

list_day=/nobackup/gamatull/dataproces/LST/geo_file/list_day.txt
awk -v nr=$nr -v  INSENS=$INSENS -v SENS=$SENS  -v DN=$DN  \'{ if (NR>nr) { print INSENS "/LST_" SENS "_akima_" DN "_day" $1 "_allobs_Fill5obs.tif" } }\' $list_day >  $RAMDIR/list_day_${SENS}_$tile.txt
awk -v INSENS=$INSENS -v SENS=$SENS -v DN=$DN \'{ if (NR<=19)   print   INSENS "/LST_" SENS "_akima_" DN "_day" $1 "_allobs_Fill5obs.tif" }\'             $list_day >> $RAMDIR/list_day_${SENS}_$tile.txt
awk -v INSENS=$INSENS -v SENS=$SENS -v DN=$DN \'{ if (NR>=20 && NR<=29 ) print  INSENS "/LST_" SENS "_akima_" DN "_day" $1 "_allobs_MskIndia.tif" }\'     $list_day >> $RAMDIR/list_day_${SENS}_$tile.txt
awk -v INSENS=$INSENS -v SENS=$SENS -v DN=$DN \'{ if (NR>=30)  print  INSENS "/LST_" SENS "_akima_" DN "_day" $1 "_allobs_Fill5obs.tif"}\'                $list_day >> $RAMDIR/list_day_${SENS}_$tile.txt
awk -v nrt=$nrt  -v  INSENS=$INSENS -v SENS=$SENS -v DN=$DN \'{ if (NR<=nrt)  print INSENS "/LST_" SENS "_akima_" DN "_day" $1 "_allobs_Fill5obs.tif"}\'  $list_day >> $RAMDIR/list_day_${SENS}_$tile.txt

gdalbuildvrt -te $geo_string -separate -overwrite -input_file_list   $RAMDIR/list_day_${SENS}_$tile.txt    $RAMDIR/LST_${SENS}_$tile.vrt 

BW=30
echo start the spline for $tile  

pkfilter -of GTiff  -ot Float32  -co COMPRESS=DEFLATE  -co ZLEVEL=9 -nodata 0  \
$( echo $(awk -v nr=$nr  \'{ if (NR>nr) {  print "-win" ,  int ($1)} }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk \'{ print "-win" ,  int ($1)+365 }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk -v nrt=$nrt   \'{ if (NR<=nrt)   print "-win" ,  int ($1)+365+365 }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ))\
 -wout 380 -fwhm $BW -wout 410 -fwhm $BW -wout 439 -fwhm $BW -wout 470 -fwhm $BW -wout 500 -fwhm $BW -wout 531 -fwhm $BW -wout 561 -fwhm $BW -wout 592 -fwhm $BW -wout 623 -fwhm $BW -wout 653 -fwhm $BW -wout 684 -fwhm $BW -wout 714 -fwhm $BW  -wout 745 -fwhm $BW    -i  $RAMDIR/LST_${SENS}_$tile.vrt   -o $RAMDIR/LST_${SENS}_spline_$tile.tif -interp akima_periodic

# validation procedure .  se si campbia il bw controllareil nr 
# gdal_translate $RAMDIR/LST_${SENS}_$tile.vrt  $RAMDIR/LST_${SENS}_$tile.tif 

# paste <(awk \'{ print  int ($1) }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk \'{ print  int ($1)+365 }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt ; awk \'{ print  int ($1)+365+365 }\' /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt  ; echo  1096 )      <(gdallocationinfo   -valonly  $RAMDIR/LST_MYD_h00v00.tif   100 100 )   > full_serie.txt 

# paste <(echo  380 ; echo  410 ; echo 439 ; echo 470 ; echo 500 ; echo 531 ; echo 561 ; echo 592 ; echo 623 ; echo 653 ; echo 684 ; echo 714 ; echo 745) <( gdallocationinfo  -valonly $RAMDIR/LST_MYD_spline_h00v00.tif 100 100   ) > month_serie.txt 

# head -1  month_serie.txt
# tail -1  month_serie.txt

mv  $RAMDIR/LST_${SENS}_spline_$tile.tif    $OUTSENS/LST_${SENS}_${DN}_spline_$tile.tif 
rm -f  $RAMDIR/LST_${SENS}_spline_$tile.tif   $RAMDIR/LST_${SENS}_$tile.vrt 

' _ 

cleanram 

gdalbuildvrt  -overwrite   $RAMDIR/LST_${SENS}_spline.vrt   $OUTSENS/LST_${SENS}_${DN}_spline_*.tif 

seq 1 13  | xargs -n 1 -P  13 bash -c $'  

band=$1 

echo start the merging action for month $band

gdal_translate  -a_nodata -1  -b $band  -co COMPRESS=LZW -co ZLEVEL=9   $RAMDIR/LST_${SENS}_spline.vrt  ${OUTSENS}_merge/LST_${SENS}_${DN}_spline_month$band.tif 

# rmosso questo mask 
# -m /nobackup/gamatull/dataproces/LST/${SENS}11A2_mean_msk/${SENS}_LST3k_mask_daySUM_wgs84_allobs_mask5noobs.tif   -msknodata 1     -nodata -1 \

pksetmask -of GTiff -co COMPRESS=LZW -co ZLEVEL=9 \
-m $MSK/${SENS}_LST3k_count_yessea_ct.tif                            -msknodata 255   -nodata -1 \
-m $MSK/${SENS}_LST3k_count_yessea_ct.tif                            -msknodata 0     -nodata -1 \
-i ${OUTSENS}_merge/LST_${SENS}_${DN}_spline_month$band.tif   -o ${OUTSENS}_merge/LST_${SENS}_${DN}_spline_month${band}_tmp.tif 

# rimosso
# -m $MSK/${SENS}_LST3k_mask_daySUM_wgs84_${DN}_allobs_mask5noobs.tif  -msknodata 1     -nodata -1 \

mv ${OUTSENS}_merge/LST_${SENS}_${DN}_spline_month${band}_tmp.tif   ${OUTSENS}_merge/LST_${SENS}_${DN}_spline_month${band}.tif 

gdalwarp -srcnodata -1 -dstnodata -1 -t_srs EPSG:4326 -r average -te -180 -90 +180 +90 -tr 0.08333333333333 0.08333333333333 -multi -overwrite  -co  COMPRESS=LZW -co ZLEVEL=9   ${OUTSENS}_merge/LST_${SENS}_${DN}_spline_month$band.tif ${OUTSENS}_merge/LST_${SENS}_${DN}_spline_month${band}_10km.tif  

' _

cleanram 

qsub -v SENS=$SENS,DN=$DN  /u/gamatull/scripts/LST/spline/sc10_kelvin2celsius_MOYD11A2.sh

