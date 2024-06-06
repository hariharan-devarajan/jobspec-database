# bash /home6/gamatull/scripts/LST/mask/sc5_mask_the_countCumulatLSTobs_MOYD11A2.sh 

# ad mask path to the cumulative observation 

#PBS -S /bin/bash
#PBS -q normal 
#PBS -l select=1:ncpus=1
#PBS -l walltime=8:00:00
#PBS -V
#PBS -o   /nobackup/gamatull/stdout
#PBS -e   /nobackup/gamatull/stderr


export DIR=/nobackupp8/gamatull/dataproces/LST

export RAMDIR=/dev/shm

# eventualmente cambiare LST con OBS 

rm /dev/shm/*.tif 


export SENS=$SENS
export DN=$DN

echo MYD Day MYD Nig  MOD Day MOD Nig | xargs -n 2 -P 4 bash -c $' 
SENS=$1
DN=$2
pksetmask -co COMPRESS=LZW -co ZLEVEL=9 -i $DIR/${SENS}11A2_mean_msk/cumulative/${SENS}_${DN}_sumCumulative.tif -m /nobackupp8/gamatull/dataproces/LST/geo_file/shp/mask_path_modis_lst.tif   -msknodata 1 -nodata 0  -o $DIR/${SENS}11A2_mean_msk/cumulative/${SENS}_${DN}_sumCumulativeNoPath.tif 
'  _ 

# /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean_msk/cumulative/MYD_Day_sumCumulativeNoPath.tif ha la zonoa intorno al mare in madagascar labeled come 0 viene esclusa dal mask 

# /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/wgs84/OBS_MYD_QC_day145_wgs84_Day.tif 
# /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/wgs84/OBS_MYD_QC_day129_wgs84_Day.tif 
#  cp   /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/old/OBS_MYD_QC_day129_wgs84.tif /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/wgs84/OBS_MYD_QC_day129_wgs84_Day.tif
# cp  /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/old/LST_MYD_QC_day129_wgs84.tif /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/wgs84/LST_MYD_QC_day129_wgs84_Day.tif 

#  cp  /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/old/LST_MYD_QC_day145_wgs84.tif /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/wgs84/LST_MYD_QC_day145_wgs84_Day.tif 
#  cp  /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/old/OBS_MYD_QC_day145_wgs84.tif /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean/wgs84/OBS_MYD_QC_day145_wgs84_Day.tif 

gdalbuildvrt -overwrite  -separate -o /dev/shm/vrt.vrt   /nobackupp8/gamatull/dataproces/LST/???11A2_mean_msk/cumulative/???_???_sumCumulativeNoPath.tif

oft-calc  -ot UInt16   /dev/shm/vrt.vrt    $RAMDIR/MYDMOD_DayNigsumCumulativeNoPath.tif <<EOF
1
#1 #2 #3 #4 + + + 
EOF

gdal_translate  -co COMPRESS=LZW -co ZLEVEL=9  $RAMDIR/MYDMOD_DayNigsumCumulativeNoPath.tif   $DIR/MYD11A2_mean_msk/cumulative/MYDMOD_DayNigsumCumulativeNoPath.tif 

# sea and areas with no ls in any day nigh MYD MOD

pkgetmask -min 1 -max 50000 -data 1 -nodata 0  -ot Byte -co COMPRESS=LZW -co ZLEVEL=9 -i $RAMDIR/MYDMOD_DayNigsumCumulativeNoPath.tif -o $DIR/MYD11A2_mean_msk/cumulative/MYDMOD_DayNigsumCumulativeNoPath_0LST.tif
rm $RAMDIR/MYDMOD_DayNigsumCumulativeNoPath.tif 