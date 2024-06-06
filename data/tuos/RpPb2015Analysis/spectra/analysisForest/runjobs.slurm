#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-20:00:00
#SBATCH --mem-per-cpu=4G

echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc491
#export X509_USER_PROXY=/home/tuos/x509up_u126986
cd /home/tuos/forest/feb02/CMSSW_7_5_8/src
eval `scramv1 runtime -sh`
cd /home/tuos/forest/feb02/CMSSW_7_5_8/src/HeavyIonsAnalysis/JetAnalysis/test/crab/mc/pu1/ana/pu1nVz1VZD

root -l -b loopFiles.C

echo "Done!"

exit 0

