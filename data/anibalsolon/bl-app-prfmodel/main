#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l vmem=50gb
#PBS -l walltime=00:30:00

set -xe

BOLD=$(jq -r .bold config.json)
EVENTS=$(jq -r '.events // ""' config.json)
STIM=$(jq -r .stim config.json)

SUB=$(jq -r '._inputs[0].meta.subject' config.json)
SES=$(jq -r '._inputs[0].meta.session // "1"' config.json)

echo $SUB
echo $SES

rm -Rf output
mkdir output

CPUS=$(grep 'cpu cores' /proc/cpuinfo | uniq | sed -re 's/cpu cores\s*:\s*([0-9]+)/\1/')
CMD="time singularity run docker://anibalsolon/bl-app-prfmodel:v0.0.1"

$CMD \
3dDeconvolve -nodata 18 1.5 -polort -1 \
    -num_stimts 1 -stim_times 1 '1D:0' SPMG1 \
    -x1D output/conv.ref.spmg1.1D

$CMD \
3dTstat -mean -prefix output/bold_mean.nii.gz $BOLD

$CMD \
3dAutomask -prefix output/bold_automask.nii.gz $BOLD

$CMD \
3dcalc -a $BOLD -b output/bold_mean.nii.gz -c output/bold_automask.nii.gz \
    -expr '100*c*(a-b)' -prefix output/bold_demean.nii.gz

export AFNI_CONVMODEL_REF=output/conv.ref.spmg1.1D
export AFNI_MODEL_PRF_STIM_DSET=$STIM
export AFNI_MODEL_PRF_ON_GRID=YES
export AFNI_MODEL_DEBUG=2

$CMD \
3dNLfim -input output/bold_demean.nii.gz \
  -mask output/bold_automask.nii.gz \
  -noise Zero \
  -signal Conv_PRF \
  -sconstr 0 -10.0 10.0 \
  -sconstr 1 -1.0 1.0 \
  -sconstr 2 -1.0 1.0 \
  -sconstr 3 0.0 1.0 \
  -SIMPLEX \
  -nrand 10000 \
  -nbest 5 \
  -bucket 0 output/Buck.PRF.nii.gz \
  -snfit output/snfit.PRF.nii.gz \
  -jobs $CPUS

$CMD \
3dcalc -a output/Buck.PRF.nii.gz'[1]' -b output/Buck.PRF.nii.gz'[2]' -expr 'sqrt(a^2+b^2)' -prefix output/polor.mag.nii.gz

$CMD \
3dcalc -a output/Buck.PRF.nii.gz'[1]' -b output/Buck.PRF.nii.gz'[2]' -expr 'atan2(-b,a)' -prefix output/polor.phase.nii.gz
