#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-40:00:00

echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc491
cd /scratch/tuos/ebeAnalysis/pbpb2015/CMSSW_7_5_8_patch4/src
#cmsenv
eval `scramv1 runtime -sh`
cd /home/tuos/tmp/tmp/newSTEG

root -l proSTEGvn.C

echo "Done!"

exit 0


