# Lxplus Batch Job Script
#set CMSSW_PROJECT_SRC="cmssw_projects/13X/cmssw131hlt6/src"
#set CFG_FILE="cfgs/steps2_3_4_5.cfg"
#set OUTPUT_FILE="Analyzer_Output.root"
#set TOP="$PWD"

#cd /afs/cern.ch/user/s/ssimon/$CMSSW_PROJECT_SRC
#eval `scramv1 runtime -csh`
#cd $TOP
#cmsRun /afs/cern.ch/user/s/ssimon/$CMSSW_PROJECT_SRC/$CFG_FILE
#rfcp Analyzer_Output.root /castor/cern.ch/user/s/ssimon/$OUTPUT_FILE

#set OUTPUT_FILE="Results.root"
#FileName1= $1
#wc $FileName1
#source /afs/cern.ch/sw/lcg/external/gcc/4.3.2/x86_64-slc5/setup.sh
#cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.03/x86_64-slc5-gcc43-opt/root/bin/
#source /afs/cern.ch/sw/lcg/external/gcc/4.7/x86_64-slc5/setup.sh
#cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.11/x86_64-slc5-gcc47-opt/root/bin/
#source /afs/cern.ch/sw/lcg/contrib/gcc/4.6/x86_64-slc6/setup.sh
#cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.09/x86_64-slc6-gcc46-opt/root/bin/
#source thisroot.sh
#export $PYTHONHOME=/usr/bin/python2.6
#cd /afs/cern.ch/work/h/hbehnami/WPolar/CMSSW_7_0_4/src
#cmsenv
#source /afs/cern.ch/sw/lcg/contrib/gcc/4.6/x86_64-slc6/setup.sh
#cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.09/x86_64-slc6-gcc46-opt/root/bin/
#source thisroot.sh
#export PATH=/afs/cern.ch/work/h/hbehnami/WPolar/LHAPDF/local/bin:$PATH
#export LD_LIBRARY_PATH=/afs/cern.ch/work/h/hbehnami/WPolar/LHAPDF/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/afs/cern.ch/work/h/hbehnami/WPolar/LHAPDF/local/lib64/python2.6/site-packages:$PYTHONPATH
#export LHAPDF_DATA_PATH=/afs/cern.ch/work/h/hbehnami/WPolar/LHAPDF/local/share/LHAPDF:$LHAPDF_DATA_PATH
#export LHAPDF_DATA_PREFIX=/afs/cern.ch/work/h/hbehnami/WPolar/LHAPDF/local/share/LHAPDF:$LHAPDF_DATA_PREFIX
#/cvmfs/cms.cern.ch/slc6_amd64_gcc481/cms/cmssw/CMSSW_7_0_4/lib/slc6_amd64_gcc481/libFWCoreServiceRegistry.so

cd /afs/cern.ch/work/h/hbehnami/WPolar/CMSSW_7_0_4/src/SimG4CMS/PPS/test 
cmsRun PPSTiming_prodRPinelasticBeta90Energy6500GeV_cfg.py
#bsub -q 8nh -J TbarW < TbarW.sh
