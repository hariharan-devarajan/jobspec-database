#!/bin/bash

##=======================================================================
#PBS -N gnu-par
#PBS -A P54048000 
#PBS -l walltime=06:00:00
#PBS -q premium
#PBS -k oe
#PBS -m a 
#PBS -M zarzycki@ucar.edu
#PBS -l select=1:ncpus=36:mem=109GB
################################################################

# use numcores = 32 for 1deg runs, 16 for high-res runs
NUMCORES=32
TIMESTAMP=`date +%s%N`
COMMANDFILE=commands.${TIMESTAMP}.txt

WGTFILE=/glade/u/home/zarzycki/work/ASD2017_files/offline-remap/map_ne30_to_1x1glob_patch.nc
CAMDIR=/glade/p/nsc/nacm0003/cmz-tmp/archive/f.asd2017.cesm20b05.FAMIPC6CLM5.ne30_g16.exp212/atm/hist/
#WGTFILE=/glade/u/home/zarzycki/work/ASD2017_files/offline-remap/map_conus_30_x8_to_0.125x0.125_GLOB.nc
#CAMDIR=/glade/p/nsc/nacm0003/cmz-tmp/archive/f.asd2017.cesm20b05.FAMIPC6CLM5.ne0conus30x8_t12.exp211/atm/hist/
#WGTFILE=/glade/u/home/zarzycki/work/ASD2017_files/offline-remap/map_mp120a_to_1x1glob_patch.nc
#CAMDIR=/glade/u/home/zarzycki/scratch/ASD/archive/f.asd2017.cesm20b05.FAMIPC6CLM5.mp120a_g16.exp214/atm/hist/
#WGTFILE=/glade/u/home/zarzycki/work/ASD2017_files/offline-remap/map_mp15a-120a-US_to_0.125x0.125glob_patch.nc
#CAMDIR=/glade/u/home/zarzycki/scratch/ASD/archive/f.asd2017.cesm20b05.FAMIPC6CLM5.mp15a-120a-US_t12.exp213/atm/hist/

OUTDIR=/glade/scratch/zarzycki/TEST-AR/

mkdir -p ${OUTDIR}
FILES=`ls ${CAMDIR}/*.cam.h2*.nc`
for f in $FILES
do
  base=`basename $f`
  LINECOMMAND="ncks -v date,U850,V850,PRECT,TMQ -d time,,,1 ${f} ${OUTDIR}/${base}_tmp1.nc ; ncremap -i ${OUTDIR}/${base}_tmp1.nc -o ${OUTDIR}/${base}_tmp2.nc -m ${WGTFILE} ; mv ${OUTDIR}/${base}_tmp2.nc ${OUTDIR}/${base} ; rm ${OUTDIR}/${base}_tmp*.nc"
  echo ${LINECOMMAND} >> ${COMMANDFILE}
done

#### Use this for Cheyenne batch jobs
parallel --jobs ${NUMCORES} -u --sshloginfile $PBS_NODEFILE --workdir $PWD < ${COMMANDFILE}

rm ${COMMANDFILE}

