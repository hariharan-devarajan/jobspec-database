
# for SENS in MYD MOD ; do  qsub -v SENS=$SENS  /u/gamatull/scripts/LST/sc3_cloudpixelcount_MOYD11A2.sh   ; done

#PBS -S /bin/bash
#PBS -q normal
#PBS -l select=1:ncpus=1
#PBS -l walltime=4:00:00
#PBS -V
#PBS -o /nobackup/gamatull/stdout
#PBS -e /nobackup/gamatull/stderr

echo /u/gamatull/scripts/LST/sc3_pixelcount_MOYD11A2.sh

export  SENS=$SENS

export INSENS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk
export OUTSENS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk/txt

export RAMDIR=/dev/shm


cat /nobackup/gamatull/dataproces/LST/geo_file/list_day.txt   | xargs -n 1 -P 46  bash -c $' 
day=$1

gdal_edit.py  -a_nodata -1   $INSENS/${SENS}_CLOUD3k_day${day}_wgs84.tif
pkinfo -hist -i  $INSENS/${SENS}_CLOUD3k_day${day}_wgs84.tif |  awk -v day=$day \'{ if (NR==1) {   printf ("%i " , day ) ; printf ("%i " , $2) } else {   printf ("%i\\n" , $2) }  }\'

' _   | sort -k 1,1 -g  >  $OUTSENS/count_CLOUD3k.txt

