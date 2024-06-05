#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-20:00:00

echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc630
cd /home/tuos/i_ana2021/CMSSW_10_0_3/src
#cmsenv
eval `scramv1 runtime -sh`

cd /store/user/tuos/forSayan

bash merging.sh listFinal.txt
#bash merging.sh testFinal.txt

echo "Done!"

exit 0

#sbatch -J sayanMerge -o output.out submit.slurm


