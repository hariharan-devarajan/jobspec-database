#!/bin/bash
#PBS -l ncpus=1
#PBS -l vmem=1000mb
#PBS -l cput=02:00:00
#PBS -l walltime=12:00:00
#PBS -q ihep-short
#PBS -V

# ^^^ qbatch system requirements.
# This is macro to run 1 thread of simulation. 
# It supposed not to be run directly by user but by other higher-level macro.
# It requires some environment variables to be setted up.
# However  

# Read config of the simulation
THIS_THREAD_PATH=$PBS_O_WORKDIR
if [ ! -z $THIS_THREAD_PATH ] ; then
    if ls $THIS_THREAD_PATH/shell_env.sh >& /dev/null ; then
	for var in WD MCDIR ANCHORS SUFFIX CONTROL PRODUCTION_NAME UNIC_CODE LD_LIBRARY_PATH FILELIST CONVERT_ONLY SEED EXTARGET PERIOD HYCONDITION NTHREADS PATH MESON PERIOD_PRFX PWD EVENTNUMBER TGT_PRFX MACRODIR PROBABILITY_SA TARGET MCRUNSDIR THICKNESS_S4 REGGEN_CARDS PROBABILITY_S4 PROBABILITY_TARG PROBABILITY_SA DELTA_S4 DELTA_SA THICKNESS_S4 DELTA_TARGET H2S_TABLE
	do
	    for val in $(grep "$var=" $THIS_THREAD_PATH/shell_env.sh) ; do
		if [ ! -z $val ] ; then 
		    export $val
		    echo 'Using earlier defined variable ' $val
		fi
	    done
	done
    fi
fi
######################################################################################################
# check if you setted up simulation configuration correctly
# 
# default values for simulation config if not setted up 
if [ -z $EXTARGET ];    then  export EXTARGET=-1;     fi; 
if [ -z $EXRESON ] ;    then  export EXRESON=-1;      fi;
if [ -z $EXCHANEL ];    then  export EXCHANEL=-1;     fi;
if [ -z $CONTROL ];     then  export CONTROL=0;       fi;
if [ -z $SEED ];        then  export SEED=0;          fi;
#$(cat /dev/urandom | tr -dc '0-9' | head --bytes 8) ;  fi; # true random 
if [ -z $TARGET ];      then  export TARGET=2;        fi;
if [ -z $EVENTNUMBER ]; then  export EVENTNUMBER=100; fi;
#
# without those variables simulation is not possible
if [ -z $WD ];          then  
    echo 'you are doing wrong!';  
    echo 'export WD=/path/to/production/directory/'
    echo 'first! then try again.';
    return; fi;
if [ -z $MACRODIR ];    then  
    echo 'you are doing wrong!';  
    echo 'export MACRODIR=/directory/with/running/macros/'
    echo 'first! then try again.';  
    return; fi;
if [ -z $FILELIST ];    then  
    echo 'you are doing wrong!';  
    echo 'export FILELIST=/path/to/filelist.dat'
    echo 'first! then try again.';                  
    return; fi;
if [ -z $SUFFIX ];      then  
    echo 'you are doing wrong!';  
    echo 'export SUFFIX=|name of directory where to generate results|'
    echo 'first! then try again.';
    return; fi;
if [ -z $ANCHORS ];     then  
    echo 'you are doing wrong!';  
    echo 'export ANCHORS=/directory/where/you/store/' 
    echo 'calibr.cards h_s_new.dat e_cor_matrix.dat coeff_old.dat' 
    echo 'first! then try again.'
    return; fi;
if [ -z $REGGEN_CARDS ];then  
    echo 'you are doing wrong!';  
    echo 'export REGGEN_CARDS=/path/to/RegGen.cards first';  
    return; fi;
######################################################################################################

#simulation summary
echo 'current WD='$WD
echo 'macro dir is ' $MACRODIR
echo 'suffix which will be used as new directory name is ' $SUFFIX
echo 'filelist to be used: ' $(ls $FILELIST)
echo 'calibr.cards to be used: ' $(ls $ANCHORS/calibr.cards)
echo 'RegGen.cards to be used: ' $(ls $REGGEN_CARDS)
echo 'number of events to be generated ' $EVENTNUMBER
echo 'target medium code is ' $TARGET
echo 'exclusive target is ' $EXTARGET
echo 'exclusive reson is ' $EXRESON
echo 'exclusive chanel is ' $EXCHANEL
echo 'generator control parameter is ' $CONTROL 
echo 'seed = ' $SEED
echo 'starting simulation...'

# VMC environment
source $MACRODIR/vmc_env.sh
LD_LIBRARY_PATH=$MCDIR/lib/tgt_linuxx8664gcc/:$LD_LIBRARY_PATH

mkdir -p $WD/$SUFFIX
cd $WD/$SUFFIX
echo "I cd'ed into " $(pwd)
# copy everything we need
echo "I remove links: "
rm -fv RegGen.cards macro file_list.dat calibr.cards h_s_new.dat coeff_old.dat e_cor_matrix.dat run_g3_control.C g3tgeoConfig.C converter
if [ ! $CONVERT_ONLY = yes ] ; then
    echo "You didn't ask only to convert MC_res.dat so I remove previous production files (if any):"
    rm -fv MC_res.dat log_converter log_production MCgen.dat Run${SEED}.gz gphysi.dat Histos.root generated.tar *.bz2
fi
ln -s $REGGEN_CARDS ./RegGen.cards 
ln -s $MACRODIR/run_g3_control.C ./run_g3_control.C
ln -s $MACRODIR/g3tgeoConfig.C ./g3tgeoConfig.C
ln -s $MACRODIR/macro/ ./macro  
ln -s $FILELIST ./file_list.dat
ln -s $ANCHORS/calibr.cards     ./calibr.cards 
ln -s $ANCHORS/h_s_new.dat	./h_s_new.dat
ln -s $ANCHORS/coeff_old.dat    ./coeff_old.dat
ln -s $ANCHORS/e_cor_matrix.dat	./e_cor_matrix.dat 
#
#cp -u $ANCHORS/calibr.cards ./
#cp -u $ANCHORS/h_s_new.dat ./
#cp -u $ANCHORS/coeff_old.dat ./
#cp -u $ANCHORS/e_cor_matrix.dat ./
# run MC priduction
if [ $CONVERT_ONLY = yes ] #if this flag is setted then we don't need to perform full simulation, only convert results to Hyperon data format
then
    #check if we have archived simulation and if so then unpack it
    if [ -e generated.tar ]; then
	tar -xf generated.tar MC_res.dat.bz2
	bzip2 -d MC_res.dat.bz2
    fi
    if [ -e MC_res.dat ] #if MC_res.dat exists then do nothing. otherwise do a full simulation in order to convert something.
    then
	echo 'you asked to skip full simulation and MC_res.dat exists... So skipping...'
    else
	echo 'you asked to skip full simulation. Hovewer MC_res.dat does not exist. So I start full simulation.'
	root -b -q run_g3_control.C\($TARGET,$EVENTNUMBER,\"\",$EXTARGET,$EXRESON,$EXCHANEL,$CONTROL,$SEED\) >& log_production; #do a full simulation
    fi
else
    root -b -q run_g3_control.C\($TARGET,$EVENTNUMBER,\"\",$EXTARGET,$EXRESON,$EXCHANEL,$CONTROL,$SEED\) >& log_production; #do a full simulation
fi
ln -s $MACRODIR/converter/exe ./converter  
if [ -f MC_res.dat ] ; then 
    echo 'I start converting MC_res.dat to' Run${SEED}.gz
    ./converter MC_res.dat $SEED >& log_converter; #convert simulation results into Hyperon data format
    # we do ln in order to save space
    # we keep only part of log_production
# We put MC_res.dat, Histos.root, MCgen.dat, log_converter into archive
    if [ -f Run${SEED}.gz ] ; then #!!! success
	echo 'File exists:' Run${SEED}.gz
	#relpath=$(realpath --relative-to=$MCRUNSDIR $(pwd))
	#ln -s $relpath/Run${SEED}.gz $MCRUNSDIR/Run${SEED}.gz
	echo 'Moving it to ' $MCRUNSDIR
	mv Run${SEED}.gz $MCRUNSDIR/
	echo 'First 700 strings and last 700 strings of log_production. For full file look MC_res.dat.bz2 ' > log_production_part
	head -n 700 log_production >> log_production_part
	echo '.................................................................' >> log_production_part
	echo '.............cutted here.........................................' >> log_production_part
	echo '.................................................................' >> log_production_part
	tail -n 700 log_production >> log_production_part
	echo 'Now I shall archive production files to generated.tar'
	if bzip2 -z9 log_production MC_res.dat MCgen.dat
	then
	    echo 'I made bz2 archives.'
	else
	    echo 'Problem to make bz2 archives.'
	fi
	if tar -cf generated.tar log_production.bz2 MC_res.dat.bz2 MCgen.dat.bz2 Histos.root 
	then
	    echo "Archive generated.tar is created. I remove files:"
	    rm -vf log_production.bz2 MC_res.dat.bz2 MCgen.dat.bz2 Histos.root 
	fi
	chmod a+r $MCRUNSDIR/Run${SEED}.gz
	chmod -R a+r .
    fi
else
    echo 'No MC_res.dat. Nothing to convert. Probably something went wrong. Please check.'
fi
