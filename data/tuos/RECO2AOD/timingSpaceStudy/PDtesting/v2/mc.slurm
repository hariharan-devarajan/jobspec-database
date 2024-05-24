#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2-20:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=job01.out
#SBATCH --job-name="MCtiming"

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc491
#cd /home/tuos/reco2AOD/CMSSW_7_4_0_pre8/src/slurm
cd /scratch/tuos/RUN/recoAOD/CMSSW_7_5_0/src/v1/runAllInOneJob
eval `scramv1 runtime -sh`

cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMBottomonia.py
cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMCharmonia.py
cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMEWQExo.py
cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMHighPtJet100
cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMHighPtLowerJetsPhotons.py
cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMHighPtPhoton40AndZ.py
cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMOniaCent30100.py
cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMOniaTnP.py
cmsRun step2_RAW2DIGI_L1Reco_MB_AODSIMUCC.py


exit 0

