#!/bin/bash
#PBS -l nodes=1:ppn=16:dc2,walltime=0:20:00
#PBS -N app-AFQclean
#PBS -V

if [ $ENV == "IUHPC" ]; then
	cd $PBS_O_WORKDIR
	module load matlab
fi

export MATLABPATH=$MATLABPATH:$SERVICE_DIR
matlab -nodisplay -nosplash -r main



if [ -s output.mat ]; 
then 
  echo 0 > finished
else
  echo "file missing"
  echo 1 > finished
  exit 1
fi
