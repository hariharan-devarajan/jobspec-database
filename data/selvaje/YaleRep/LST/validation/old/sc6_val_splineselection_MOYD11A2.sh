
#  qsub -v SENS=MOD    /u/gamatull/scripts/LST/sc6_val_splineselection_MOYD11A2.sh 
# combine the spine results and check the best graph 

#PBS -S /bin/bash
#PBS -q devel
#PBS -l select=1:ncpus=4
#PBS -l walltime=2:00:00
#PBS -V
#PBS -o /nobackup/gamatull/stdout
#PBS -e /nobackup/gamatull/stderr

echo $SENS   /u/gamatull/scripts/LST/sc5_fillspline_MOYD11A2.sh 

export  SENS=MOD

export  INSENS=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_val/${SENS}11A2_splinefill
export RAMDIR=/dev/shm


rm -f /dev/shm/*
echo merge the observation txt  
for spline in cspline akima ; do
    cat /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_splinefill/txt/LST_${SENS}_${spline}_h??v??.txt  |  awk '{ if  ($5!=0) print  }'   >    /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_splinefill/txt/LST_${SENS}_${spline}_alltile.txt
done 

echo  processing the all txt
for HOL in 1 2 3 4 5 6 7 8 9 ; do 
    echo $HOL
    OUTXT=/nobackupp8/gamatull/dataproces/LST/${SENS}11A2_val/${SENS}11A2_splinefill_${HOL}hole/txt
    echo processing the spline 
    for spline in cspline akima ; do 
	echo $spline 
	cat $OUTXT/LST_${SENS}_${spline}_h??v??.txt  |  awk '{ if  ($5!=0) print  }'   >    $OUTXT/LST_${SENS}_${spline}_alltile.txt 
  
	seq_day=$(grep " $HOL" /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_val/day_holl_numb | awk '{ print $1  }')
	echo $seq_day
	for day in $( echo $seq_day) ; do 
	    banday=$(grep "$day" /nobackupp8/gamatull/dataproces/LST/geo_file/list_day_nr.txt  | awk '{ print $2 }')
	    col=$(expr $banday + 4) 
	    echo $col $banday 
	    paste  <(awk -v col=$col  '{ print $col  }'  /nobackupp8/gamatull/dataproces/LST/${SENS}11A2_splinefill/txt/LST_${SENS}_${spline}_alltile.txt   )  <(awk -v col=$col '{ print $col }' $OUTXT/LST_${SENS}_${spline}_alltile.txt  )  >   $OUTXT/LST_${SENS}_${spline}_alltile_day$banday.txt  
	done  
	cat  $OUTXT/LST_${SENS}_${spline}_alltile_day??.txt > $OUTXT/LST_${SENS}_${spline}_alltile_dayALLday.txt
    done
done 


