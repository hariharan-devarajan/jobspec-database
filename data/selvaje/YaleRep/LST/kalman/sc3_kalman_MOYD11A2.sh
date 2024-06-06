
#  qsub -v SENS=MYD /u/gamatull/scripts/LST/sc3_kalman_MOYD11A2.sh 

#PBS -S /bin/bash
#PBS -q devel
#PBS -l select=1:ncpus=1
#PBS -l walltime=2:00:00
#PBS -V
#PBS -o /nobackup/gamatull/stdout
#PBS -e /nobackup/gamatull/stderr

echo sc2_spline_MOYD11A2 

export  SENS=MYD

export OBS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean
export MOD=/nobackupp8/gamatull/dataproces/LST/MOYD11C2
export KAL=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_kalman
export RAMDIR=/dev/shm

echo start the pkkalman

# replicate the dataset 3 times + in the end the first day.

# awk '{ if (NR>1) print $1   }' /nobackupp8/gamatull/dataproces/LST/geo_file/tile_lat_long_20d_land.txt | head -1   | xargs -n 1 -P  40  bash -c $' 

echo h24v06     | xargs -n 1 -P  1  bash -c $' 

tile=$1

geo_string=$(grep $tile /nobackupp8/gamatull/dataproces/LST/geo_file/tile_lat_long_20d.txt | awk \'{ print $4 , $5 , $6 , $7 }\')
geo_string="73.7 26.10 79.45 23.25"

# clip the tiles

# for day  in $( cat /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt  )  ; do 
# gdal_translate -co COMPRESS=LZW -co ZLEVEL=9  -projwin $geo_string       $OBS/LST_${SENS}_QC_day${day}_wgs84.tif   $RAMDIR/LST_${SENS}_QC_day${day}_$tile.tif   
# gdal_translate -co COMPRESS=LZW -co ZLEVEL=9  -projwin $geo_string -b 1  $MOD/LST_Day_CMG_day${day}.tif            $RAMDIR/LST_Day_CMG_day${day}_$tile.tif  
# done 

# gdalbuildvrt    -separate   -overwrite  $RAMDIR/LST_Day_CMG_day.vrt   $RAMDIR/LST_Day_CMG_day???_h24v06.tif
# gdal_translate   -co COMPRESS=LZW -co ZLEVEL=9  $RAMDIR/LST_Day_CMG_day.vrt  $RAMDIR/LST_Day_CMG_day.tif  
# pkfilter -of GTiff  -ot Float32  -co BIGTIFF=YES   -co COMPRESS=LZW -co ZLEVEL=9 -nodata 0  -f smoothnodata  -dz 1   -interp akima    -i  $RAMDIR/LST_Day_CMG_day.tif  -o $RAMDIR/LST_Day_CMG_smooth.tif

# cat /nobackup/gamatull/dataproces/LST/geo_file/list_day_nr.txt | xargs -n 2 -P 23 bash -c $\' 

# gdal_translate -b $2   -co COMPRESS=LZW -co ZLEVEL=9  $RAMDIR/LST_Day_CMG_smooth.tif  $RAMDIR/LST_Day_smoothCMG_day${1}_h24v06.tif

# \' _ 



echo start the kalman   

tile=h24v06

# /home6/gamatull/pktools-2.6.3/bin/pkkalman   -of GTiff  -ot Float32 -co COMPRESS=LZW -co ZLEVEL=9  -dir forward  -dir backward -dir smooth   -win 0 -down 7 -um 2  -th 1000  \
# $( ls $RAMDIR/LST_Day_smoothCMG_day???_$tile.tif       | xargs -n 1 echo -mod  ) $(awk \'{ print int($1)  }\'    /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt | xargs -n 1 echo -tmod ) \
# $( ls $RAMDIR/LST_${SENS}_QC_day???_$tile.tif    | xargs -n 1 echo -obs  ) $(awk \'{ print int($1)  }\'   /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt | xargs -n 1 echo -tobs ) \
# -modnodata 0 -obsnodata 0  -unodata 10000 -rs   -win 15 -ofw  $KAL/LST_${SENS}_${tile}_kal_ofw  -obw  $KAL/LST_${SENS}_${tile}_kal_obw   -ofb  $KAL/LST_${SENS}_${tile}_kal_ofb  

/home6/gamatull/pktools-2.6.3/bin/pkkalman   -of GTiff  -ot Float32 -co COMPRESS=LZW -co ZLEVEL=9  -dir forward  -dir backward -dir smooth   -win 0 -down 7 -um 2  -th 1000  \
$( ls $RAMDIR/LST_Day_CMG_day???_$tile.tif       | xargs -n 1 echo -mod  ) $(awk \'{ print int($1)  }\'    /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt | xargs -n 1 echo -tmod ) \
$( ls $RAMDIR/LST_${SENS}_QC_day???_$tile.tif    | xargs -n 1 echo -obs  ) $(awk \'{ print int($1)  }\'   /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt | xargs -n 1 echo -tobs ) \
-modnodata 0 -obsnodata 0  -unodata 10000 -rs   -win 15 -ofw  $KAL/LST_${SENS}_${tile}_kal_nosmooth_ofw  -obw  $KAL/LST_${SENS}_${tile}_kal_nosmooth_obw   -ofb  $KAL/LST_${SENS}_${tile}_kal_nosmooth_ofb  


cat   /proc/meminfo  | grep MemFree

' _ 







geo_string="73.7 26.10 79.45 23.25"

gdal_translate -co COMPRESS=LZW -co ZLEVEL=9  -projwin $geo_string -b 1

