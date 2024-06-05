#!/bin/bash

#INPUT Sequences file

#SBATCH -J e1d_sim
#SBATCH -n 1
#SBATCH -N 1

#SBATCH -p defq
#SBATCH --output=log/e1d_sim_%A_%a.out
#SBATCH --error=log/e1d_sim_%A_%a.err
#SBATCH --array=1-199

echoerr() { printf "%s\n" "$*" >&2; }
echoerr "Hostname: $HOSTNAME"

#change tmp
export TMPDIR="/work/tylerns/e1d_sim/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID"
export SINGULARITY_CACHEDIR="$TMPDIR/sing_cache"

#load singularity
module load singularity/2.6.0

export JOB_DIR=$PWD
export CONFIG_DIR=$PWD
export OUTPUT_DIR=/work/gothelab/clas6/e1d/sim
export DATA_DIR=/work/gothelab/clas6
export DATE=`date +%m-%d-%Y`
export TYPE="sim_e1d_maid2007_highq2_${DATE}"

# Copy in configuration files
mkdir -p $TMPDIR
mkdir -p $SINGULARITY_CACHEDIR
cp $JOB_DIR/do_sim.sh $TMPDIR/do_sim.sh
cp $JOB_DIR/aao_rad.inp $TMPDIR/aao_rad.inp
cp $JOB_DIR/gsim.inp $TMPDIR/gsim.inp
cp $JOB_DIR/user_ana.tcl $TMPDIR/user_ana.tcl
cd $TMPDIR

res1=$(date +%s.%N)
#run singularity
singularity shell -B $DATA_DIR:/data \
		  -B $TMPDIR:/code \
		  -B $DATA_DIR/parms/recsis:/recsis \
		  $DATA_DIR/clas6.img -c "cd /code && bash do_sim.sh"

xz -9 -v -T4 $TMPDIR/gsim_no_gpp.bos
#cp $TMPDIR/gsim_no_gpp.bos.xz $OUTPUT_DIR/gsim/npip/gsim_no_gpp_${TYPE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.bos.xz
cp $TMPDIR/all.root $OUTPUT_DIR/npip/${TYPE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.root

echoerr "Removing $TMPDIR from $HOSTNAME"
du -sh $TMPDIR
#rm -rf $TMPDIR

res2=$(date +%s.%N)
dt=$(echo "$res2 - $res1" | bc)
dd=$(echo "$dt/86400" | bc)
dt2=$(echo "$dt-86400*$dd" | bc)
dh=$(echo "$dt2/3600" | bc)
dt3=$(echo "$dt2-3600*$dh" | bc)
dm=$(echo "$dt3/60" | bc)
ds=$(echo "$dt3-60*$dm" | bc)
echo "Hostname: $HOSTNAME"
printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
