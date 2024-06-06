
	# qsub -v SENS=MYD  /home6/gamatull/scripts/LST/mask/sc4_countLSTobs_MOYD11A2.sh 

# create a mask that is able to identify pixel with no observation in all the temporal series ...inclued cloud areas and sea areas
# this support has to add to the mask sea 

#PBS -S /bin/bash
#PBS -q normal 
#PBS -l select=1:ncpus=1
#PBS -l walltime=8:00:00
#PBS -V
#PBS -o   /nobackup/gamatull/stdout
#PBS -e   /nobackup/gamatull/stderr

export SENS=${SENS}
export IN=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean
export OUT=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk

export RAMDIR=/dev/shm

# eventualmente cambiare LST con OBS 

rm /dev/shm/*.tif 


echo MYD Day MYD Nig MOD Day MOD Nig | xargs -n 2 -P 4 bash -c $' 
SENS=$1
DN=$2
~/bin/pkcomposite  $( ls /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean/wgs84/LST_${SENS}_QC_day???_wgs84_$DN.tif | xargs -n 1 echo -i )  -ot Byte   -co  COMPRESS=LZW -co ZLEVEL=9  -cr sum -dstnodata 0  -bndnodata 0 -file 1  -o /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk/${SENS}_LST3k_count_$DN.tif

gdal_translate -b 2 -ot Byte -co  COMPRESS=LZW -co ZLEVEL=9  /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk/${SENS}_LST3k_count.tif    /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_mean_msk/${SENS}_LST3k_count_nosea.tif

' _ 




# in this case see 255    0 value with ls but not observation 
~/pktools-2.6.3/bin/pksetmask -ot Byte  -co  COMPRESS=LZW -co ZLEVEL=9   -m /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean_msk/MYD_SEA_mask_wgs84.tif   -msknodata 1 -nodata 255 -i /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean_msk/MYD_LST3k_count_nosea.tif   -o /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean_msk/MYD_LST3k_count_yessea.tif

~/pktools-2.6.3/bin/pksetmask -ot Byte  -co  COMPRESS=LZW -co ZLEVEL=9   -m /nobackupp8/gamatull/dataproces/LST/MOD11A2_mean_msk/MOD_SEA_mask_wgs84.tif   -msknodata 1 -nodata 255 -i /nobackupp8/gamatull/dataproces/LST/MOD11A2_mean_msk/MOD_LST3k_count_nosea.tif   -o /nobackupp8/gamatull/dataproces/LST/MOD11A2_mean_msk/MOD_LST3k_count_yessea.tif 

pkcreatect -min 0 -max 46   > /tmp/color.txt
echo 255 255 204 204 >>  /tmp/color.txt

pkcreatect   -ot Byte   -co COMPRESS=LZW -co ZLEVEL=9   -ct  /tmp/color.txt  -i  /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean_msk/MYD_LST3k_count_yessea.tif -o  /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean_msk/MYD_LST3k_count_yessea_ct.tif 
pkcreatect   -ot Byte   -co COMPRESS=LZW -co ZLEVEL=9   -ct  /tmp/color.txt  -i  /nobackupp8/gamatull/dataproces/LST/MOD11A2_mean_msk/MOD_LST3k_count_yessea.tif -o  /nobackupp8/gamatull/dataproces/LST/MOD11A2_mean_msk/MOD_LST3k_count_yessea_ct.tif 

pkinfo -hist -i  /nobackupp8/gamatull/dataproces/LST/MOD11A2_mean_msk/MOD_LST3k_count_yessea_ct.tif  | grep -v " 0"  >  /nobackupp8/gamatull/dataproces/LST/MOD11A2_mean_msk/MOD_LST3k_count_yessea_ct_hist.txt
pkinfo -hist -i  /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean_msk/MYD_LST3k_count_yessea_ct.tif  | grep -v " 0"  >  /nobackupp8/gamatull/dataproces/LST/MYD11A2_mean_msk/MYD_LST3k_count_yessea_ct_hist.txt
