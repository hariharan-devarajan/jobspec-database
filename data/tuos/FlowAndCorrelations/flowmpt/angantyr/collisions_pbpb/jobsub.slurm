#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-20:00:00


echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc700

cd /scratch/tuos/trigger/CMSSW_10_3_0_pre5/src

#cmsenv
eval `scramv1 runtime -sh`

export PYTHIA8DATA=/scratch/tuos/models/angantyr/root/dev/pythia8305/share/Pythia8/xmldoc

cd /scratch/tuos/models/angantyr/root/dev/pythia8305/production/collisions_pbpb/ba100/jb0


./pbpb_angantyr_5tev batchNum jobNum

echo "Done!"

exit 0


