#!/bin/sh
#PBS -l select=1:ncpus=40:mpiprocs=40:ompthreads=1:jobtype=small
#PBS -l walltime=167:00:00
prefix=$TEMPLATE_PROJECT_PREFIX

if [ "$PBS_O_WORKDIR" ]; then
  cd ${PBS_O_WORKDIR}
fi

module purge
source $prefix/util/load_curp_ims

set -xeu

output=$prefix/5curp/output/$group_type/${run}_${smp}
# faster directory
tmp=/ramd/users/$USER/$PBS_JOBID

mkdir -p $output
cd $tmp

echo "job start time" `date +'%Y%m%d %H:%M:%S'` >> $output/time.log

cp $prefix/0structure/output/system.dry.prmtop .
cp $prefix/0structure/output/atomgroup_sidechain.dat atom_group.dat
cp $prefix/4nve/output/${run}_${smp}/center.crd.nc md.crd.nc
cp $prefix/4nve/output/${run}_${smp}/md.vel.nc .
cp $prefix/5curp/${group_type}.cfg flux.cfg
cp $prefix/5curp/output/group_pair/${group_type}.dat group_pair.dat
cp $prefix/5curp/output/group_pair/${group_type}.dat $output/group_pair_used.dat

echo "copy end time " `date +'%Y%m%d %H:%M:%S'` >> $output/time.log

mpirun -np 40 curp compute flux.cfg > $output/flux.log

echo "flux end time " `date +'%Y%m%d %H:%M:%S'` >> $output/time.log

cp flux_grp.nc $output/flux.nc

echo "copy end time " `date +'%Y%m%d %H:%M:%S'` >> $output/time.log
