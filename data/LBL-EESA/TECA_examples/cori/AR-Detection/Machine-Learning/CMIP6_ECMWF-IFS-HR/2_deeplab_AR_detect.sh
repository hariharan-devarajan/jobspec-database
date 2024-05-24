#!/bin/bash
#SBATCH -C knl
#SBATCH -N 1484
#SBATCH -q regular
#SBATCH -t 00:45:00
#SBATCH -A m1517
#SBATCH -J 2_deeplab_AR_detect

# load the gcc module
module swap PrgEnv-intel PrgEnv-gnu

# bring a TECA install into your environment
module use /global/common/software/m1517/teca/cori/develop/modulefiles
module load teca

# print the commands aas the execute, and error out if any one command fails
set -e
set -x

# 94964 steps
# 4 steps per rank
# === 23741 ranks
# 4 cores per rank
# 64 cores per node
# === 16 ranks per node
# === 1484 nodes

pytorch_model=/global/cscratch1/sd/loring/teca_testing/TECA_data/cascade_deeplab_IVT.pt

# make a directory for the output files
out_dir=HighResMIP_ECMWF_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_6hrPlevPt/deeplab_all

rm -rf ${out_dir}
mkdir -p ${out_dir}

# do the ar detections.
time srun -N 1484 -n 23741 teca_deeplab_ar_detect \
    --pytorch_model ${pytorch_model} \
    --input_file ./HighResMIP_ECMWF_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_6hrPlevPt.mcf \
    --compute_ivt --wind_u ua --wind_v va --specific_humidity hus \
    --write_ivt --write_ivt_magnitude \
    --output_file ${out_dir}/deeplab_AR_%t%.nc \
    --steps_per_file 128

