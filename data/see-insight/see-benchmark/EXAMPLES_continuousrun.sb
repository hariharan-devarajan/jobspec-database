#!/bin/bash
#SBATCH --array=1-150
#SBATCH --time=4:00:00
#SBATCH --mem=6gb
#SBATCH --job-name=SEE-continuous

#TODO add output file to use a "Continuous_Output" folder

cd see-segment; git log -n 1; cd ..

echo "Continuous Run Number $SLURM_ARRAY_TASK_ID"

# 24 hours 85800
# 4 hours 14100 

#TODO Change location of best_so_far to Continous_Output folder

timeout 14100s singularity exec -B /usr/bin/:/sysbin/ \
                 --env PATH=/miniconda3/bin/:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/sysbin/ \
                 --writable-tmpfs \
                 --env-file /opt/software/powertools/share/jupyter.env \
                 centos7.sif \
                 ./EXAMPLES_run.sh $SLURM_ARRAY_TASK_ID $(( $SLURM_ARRAY_TASK_ID + $SLURM_JOB_ID )) coninuousrun
out=$?
echo "CHECKING FOR ERRORS"

#TODO check "STOP.output" flag in continous_output folder


if [ "$out" == "124" ] #TIMEOUT REACHED
then
    echo "Timeout...RESTARTING $out"
    sbatch --array=${SLURM_ARRAY_TASK_ID} continuousrun.sb 
else
    echo "ERROR: $out - NO RESTART"
fi

