#!/bin/bash

submitdir=$1 ## path to submission directory
action=$2 ## option for: 0=create scripts, 1=submit, 2=check, ..
cfg=$3  ## only for action=0

#######
## hard coded options
jobtype=csv ##step2, step3, step4, step5 , csv ,  corr

condorqueue=testmatch  #microcentury , workday, testmatch,  local (lxplus jobs in series, not condor), # note in resubmission to change queue need to modify the .sub job file
CAF=0 # Use the T0 cluster : https://batchdocs.web.cern.ch/local/specifics/CMS_CAF_tzero.html  , need to:  module load lxbatch/tzero
MAXJOBS=1000000 #useful for testing


##Afterglow corrections for RawPCCProducer (csv) jobs 
#DBDIR=""
#DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/Commissioning2021_v2/AlCaLumiPixelsCountsExpress/step4/
#DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/Moriond2023PAS/Random_Run2Type2Params/Run2022E/
#DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/Moriond2023PAS/Random/Run2022E
#DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/SummerLUMPAS22/Random_v2/Run2022F
#DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/SummerLUMPAS22/Random_v4/Run2022G
#DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/Reprocess2023/Random
#DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/2024Data/Random/Run2024A
DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/Reprocess2023/Random/Run2023D
#DBDIR=/eos/user/b/benitezj/BRIL/PCC_Run3/Reprocess2023_v2/Random/Run2023D

brilcalc='/usr/bin/singularity -s exec  --env PYTHONPATH=/home/bril/.local/lib/python3.10/site-packages /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/brilws-docker:latest brilcalc lumi -u hz/ub --byls --output-style csv -c offline'
#brilcalc='brilcalc lumi -u hz/ub --byls --output-style csv -c offline'
#BRILCALCDATATAG=online 
BRILCALCDATATAG=23v1

#BRILCALCTYPE=hfet
#BRILCALCTYPE=pxl
BRILCALCTYPE=bcm1futca
#BRILCALCTYPE=dt

#BRILCALCNORM=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_BRIL.json
#BRILCALCNORM=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_hfet.json
#BRILCALCNORM=hfet22v10
#BRILCALCNORM=hfet23v07
#BRILCALCNORM=dt23v03
#BRILCALCNORM=dt23v04b
#BRILCALCNORM=hfoc23v03
#BRILCALCNORM=bcm1futca23v02
BRILCALCNORM=bcm1futcav07d
#BRILCALCNORM=pltzero23v5
#BRILCALCNORM=bcm1f23v02
#BRILCALCNORM=bcm1futca23v07c
#BRILCALCNORM=pcc23VdMRescaledv0
#BCIDS="282,822,1881,2453,2474,2579,2632,2653,2716,2758,2944,3123,3302"
BRILCALCREFDETNAME=$BRILCALCNORM


baseoutdir=/eos/user/b/benitezj/BRIL/PCC_Run3
#plotsdir=/afs/cern.ch/user/b/benitezj/www/BRIL/PCC_lumi/$submitdir
plotsdir=/eos/user/b/benitezj/www/plots/BRIL/PCC_lumi/$submitdir


###########################################################
### 
if [ "$submitdir" == "" ]; then
    echo "invalid submitdir"
    return 0
fi
fullsubmitdir=`readlink -f $submitdir`
echo "fullsubmitdir: $fullsubmitdir"

INSTALLATION=${CMSSW_BASE}/src
if [ "$INSTALLATION" == "/src" ]; then
    echo "invalid INSTALLATION"
    return 0
fi
echo "INSTALLATION: $INSTALLATION"

outputdir=$baseoutdir/$submitdir
echo "output: $outputdir"

echo "condor queue: $condorqueue"

echo "job type: $jobtype"

echo "plots dir: $plotsdir"

echo "afterglow corrections: $DBDIR $DBROOTDIR"


#####################################################
## copy the cfg
if [ "$action" == "0" ]; then 
    if [ "$cfg" == "" ]; then
	echo "No cfg provided\n"
	return 0
    fi
    /bin/cp $cfg $fullsubmitdir/cfg.py
    echo "mkdir -p $outputdir"
    mkdir -p $outputdir
    /bin/ls $outputdir

fi

## clean up the plots
if [ "$action" == "5" ] ; then
    rm -rf $plotsdir
    mkdir -p $plotsdir
fi


##################################################
#### helper functions
submit(){
    local run=$1
    rm -f $fullsubmitdir/${run}.log
    rm -f ${outputdir}/${run}.root
    rm -f ${outputdir}/${run}.db
    rm -f ${outputdir}/${run}.txt
    rm -f ${outputdir}/${run}.csv
    rm -f ${outputdir}/${run}.mod
    rm -f ${outputdir}/${run}.frac

    #bsub -q 1nd -o $fullsubmitdir/${run}.log -J $run < $fullsubmitdir/${run}.sh    
    if [ "$condorqueue" == "local" ] ; then
	source $fullsubmitdir/${run}.sh
    else
	condor_submit $fullsubmitdir/${run}.sub 
    fi
}

make_sh_script(){
    local run=$1

    rm -f $fullsubmitdir/${run}.sh

    echo "export X509_USER_PROXY=${HOME}/x509up_u55361 " >> $fullsubmitdir/${run}.sh
    echo "cd ${INSTALLATION} " >> $fullsubmitdir/${run}.sh
    echo "eval \`scramv1 runtime -sh\` " >> $fullsubmitdir/${run}.sh
    echo "cd \$TMPDIR  "   >> $fullsubmitdir/${run}.sh
    echo "pwd  "   >> $fullsubmitdir/${run}.sh

    echo "export INPUTFILE=${fullsubmitdir}/${run}.txt" >> $fullsubmitdir/${run}.sh

    ###if DBDIR is set this will override the Afterglow corrections 
    if [ "$DBDIR" != "" ]; then
	echo "export DBFILE=${DBDIR}/${run}.db" >> $fullsubmitdir/${run}.sh
    fi
    if [ "$DBROOTDIR" != "" ]; then
	echo "export DBROOT=${DBROOTDIR}/${run}.root" >> $fullsubmitdir/${run}.sh
    fi
    
    ###if run.json file is there, this will apply a json when processing  
    if [ -f ${fullsubmitdir}/${run}.json ]; then
	echo "export JSONFILE=${fullsubmitdir}/${run}.json" >> $fullsubmitdir/${run}.sh
    fi
    
    ##print all environment into log file right before executing
    echo "env" >> $fullsubmitdir/${run}.sh

    ### cmsRun 
    echo "cmsRun  ${fullsubmitdir}/cfg.py" >> $fullsubmitdir/${run}.sh

    ###product that should be saved to output
    if [ "$jobtype" == "step2" ] ; then
	echo "cp step2_ALCAPRODUCER.root $outputdir/${run}.root " >> $fullsubmitdir/${run}.sh
    fi

    if [ "$jobtype" == "step3" ] ; then
	echo "cp PromptCalibProdLumiPCC.root $outputdir/${run}.root " >> $fullsubmitdir/${run}.sh
    fi
    
    if [ "$jobtype" == "step4" ] ; then
	echo "cp promptCalibConditions.db $outputdir/${run}.db " >> $fullsubmitdir/${run}.sh
	echo "cp CorrectionHisto.root $outputdir/${run}.root " >> $fullsubmitdir/${run}.sh
    fi

    if [ "$jobtype" == "step5" ] ; then
	echo "cp step5_onlyRawPCCProducer_ALCAPRODUCER.root  $outputdir/${run}.root " >> $fullsubmitdir/${run}.sh
    fi

    if [ "$jobtype" == "csv" ] ; then
	echo "cp rawPCC.csv  $outputdir/${run}.csv " >> $fullsubmitdir/${run}.sh
    fi

    ###
    if [ "$jobtype" == "corr" ] ; then
        echo "cp PCC_Corr.db  $outputdir/${run}.db " >> $fullsubmitdir/${run}.sh
        echo "cp CorrectionHisto.root  $outputdir/${run}.root " >> $fullsubmitdir/${run}.sh
    fi

}    
    

make_sub_script(){
    local run=$1
    rm -f $fullsubmitdir/${run}.sub

    echo "Universe   = vanilla" >>  $fullsubmitdir/${run}.sub
    echo "+JobFlavour = \"${condorqueue}\" " >> $fullsubmitdir/${run}.sub
    if [ $CAF == 1 ] ; then echo "+AccountingGroup = \"group_u_CMS.CAF.COMM\"" >> $fullsubmitdir/${run}.sub ; fi
    echo "Executable = /bin/bash" >> $fullsubmitdir/${run}.sub 
    echo "Arguments  = ${fullsubmitdir}/${run}.sh" >> $fullsubmitdir/${run}.sub 
    echo "Log        = ${fullsubmitdir}/${run}.log" >> $fullsubmitdir/${run}.sub 
    echo "Output     = ${fullsubmitdir}/${run}.log" >> $fullsubmitdir/${run}.sub 
    echo "Error      = ${fullsubmitdir}/${run}.log" >> $fullsubmitdir/${run}.sub 
    echo "Queue  " >> $fullsubmitdir/${run}.sub 

}


check_log(){
    local run=$1
    fail=0
    
#    if [ ! -f $fullsubmitdir/${run}.log ]; then
#	echo "no log"
#	fail=1
#    fi
    
#    if [ "$fail" == "0" ]; then
#	success=`cat $fullsubmitdir/${run}.log | grep "Normal termination"`
#	if [ "$success" == "" ]; then
#	    echo "no Success"
#	    fail=1
#	fi
#    fi
    
#    if [ "$fail" == "0" ]; then
#	fatal=`cat $fullsubmitdir/${run}.log | grep Fatal`
#	if [ "$fatal" != "" ]; then
#	    echo "Fatal"
#	    fail=1
#	fi
#    fi
    
    
    ### error check for lumi jobs	
#    if [ "$jobtype" == "lumi" ] && [ "$fail" == "0" ]; then
#	if [  ! -f  $outputdir/${run}.csv ]; then
#	    echo "no csv"
#	    fail=1
#	fi
#    fi
    
    
    ### error check for corrections jobs
    if [ "$jobtype" == "step4" ] && [ "$fail" == "0" ] ; then
	if [ ! -f $outputdir/${run}.db ]; then
	    echo "no db"
	    fail=1
	fi
    fi

    ### error check for corrections jobs
    if [ "$jobtype" == "csv" ] && [ "$fail" == "0" ] ; then
	if [ ! -f $outputdir/${run}.csv ]; then
	    echo "no csv"
	    fail=1
	fi
    fi


    ### error check for corrections jobs
    if [ "$jobtype" == "corr" ] && [ "$fail" == "0" ] ; then
        if [ ! -f $outputdir/${run}.db ]; then
            echo "no db"
            fail=1
        fi
    fi
    
}



###################################################################
##### loop over the runs
##################################################################
export RUNLIST=""
counter=0
counterbad=0
for f in `/bin/ls $fullsubmitdir | grep .txt | grep -v "~" `; do
    if [[ $counter -gt 100000 ]]; then
	 break
    fi

    run=`echo $f | awk -F".txt" '{print $1}'`
    #echo $run
    RUNLIST=$RUNLIST,$run

    ##create the scripts
    if [ "$action" == "0" ]; then
	make_sh_script $run
	make_sub_script $run
    fi

    ##submit to lxbatch
    if [ "$action" == "1" ]; then
	submit $run
    fi

    ##check failed jobs
    if [ "$action" == "2" ] ; then
	check_log $run
	if [ "$fail" == "1" ]; then
	    echo $fullsubmitdir/${run}.log
	    cat $fullsubmitdir/${run}.log | grep sqlite_file
	    counterbad=`echo $counterbad | awk '{print $1+1}'`

	    ##replace queue
	    #`sed -i 's/workday/testmatch/g' $fullsubmitdir/${run}.sub` 
	fi   
    fi

    if [ "$action" == "3" ]; then
	check_log $run
	if [ "$fail" == "1" ]; then
	    submit $run
	fi   
    fi


    ##run plotting scripts
    if [ "$action" == "4" ] ; then
	command="${brilcalc} -r ${run}"
	if [ ! -z "$BRILCALCDATATAG" ] ; then
	    command="${command} --datatag ${BRILCALCDATATAG}"
	fi
	if [ ! -z "$BRILCALCTYPE" ] ; then
	    command="${command} --type ${BRILCALCTYPE}"
	fi
	if [ ! -z "$BRILCALCNORM" ] ; then
	    command="${command} --normtag ${BRILCALCNORM}"
	fi
	if [ ! -z "$BCIDS" ] ; then
	    command="${command} --xing --xingId ${BCIDS}"
	fi
        echo $command
        #${command} | grep ${run}: | sed -e 's/,/ /g' | sed -e 's/:/ /g' | sed -e 's/\[//g'  | sed -e 's/\]//g' > $outputdir/${run}.ref
	${command} | grep ${run}:  > $outputdir/${run}.ref
    fi 

    counter=`echo $counter | awk '{print $1+1}'`
done
echo "Total runs: $counter"
echo "Total bad runs: $counterbad"

echo ${RUNLIST:1}


if [ "$action" == "5" ] ; then
    root -b -q -l ${INSTALLATION}/BRILAnalysisCode/PCCAnalysis/plots/rootlogon.C  ${INSTALLATION}/BRILAnalysisCode/PCCAnalysis/plots/plotCSVList.C\(\"${outputdir}\",\"${plotsdir}\",\"${RUNLIST:1}\",\"${BRILCALCREFDETNAME}\"\)
fi

if [ "$action" == "6" ] ; then
    NLS=`cat ${plotsdir}/ls.dat | wc -l`
    root -b -q -l ${INSTALLATION}/BRILAnalysisCode/PCCAnalysis/plots/rootlogon.C ${INSTALLATION}/BRILAnalysisCode/PCCAnalysis/plots/plotPCCStability.C\(\"${plotsdir}\",0,${NLS},\"${BRILCALCREFDETNAME}\"\)
    #root -b -q -l ${INSTALLATION}/BRILAnalysisCode/PCCAnalysis/plots/plotPCCruns.C\(\"${plotsdir}\"\)
fi

if [ "$action" == "7" ] ; then
    root -b -q -l ${INSTALLATION}/BRILAnalysisCode/PCCAnalysis/plots/rootlogon.C ${INSTALLATION}/BRILAnalysisCode/PCCAnalysis/plots/plot_afterglow_residual.C\(\"${outputdir}\",\"${plotsdir}\",\"${RUNLIST:1}\"\)
fi
