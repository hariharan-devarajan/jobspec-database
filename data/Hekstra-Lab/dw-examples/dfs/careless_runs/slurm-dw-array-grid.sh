#!/bin/bash
#SBATCH --job-name=gpu_careless_reduce
#SBATCH -p gpu_requeue,seas_gpu # partition (queue)
#SBATCH --mem 90G # memory pool for all cores
#SBATCH -t 0-03:00 # time (D-HH:MM)
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --array=11 #2-10
#SBATCH -o myoutput_%j.out
#SBATCH -e myoutput_%j.err

#  ** PROCESS GRID PARAMETERS **
PARAM_FILE=slurm_params.txt
#IL	MLPL    ITER    STDOF   PEF     rDW	RU
#MY_PARAMS=$(sed "${0}q;d" ${PARAM_FILE})
#echo $MY_PARAMS   
MY_PARAMS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${PARAM_FILE})
TMP=(${MY_PARAMS///})
echo "Parameters from slurm_params.txt: ${MY_PARAMS}"
echo $TMP
IL="${TMP[0]}"
ITER="${TMP[1]}"
STDOF="${TMP[2]}"
PEF="${TMP[3]}"
R="${TMP[4]}"
RU="${TMP[5]}"

MODE="mono"
DMIN=1.0
TEST_FRACTION=0.1
SEED=$RANDOM
BASENAME=dfs
HALF_REPEATS=3


INPUT_MTZS=(
  ../20221007_unscaled_unmerged/reference/out_corrected_root2_ohp.mtz \
  ../20221007_unscaled_unmerged/reference/out_corrected_root2_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0115/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0116/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0123/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0124/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0131/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0132/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0137/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0138/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0139/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0142/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0148/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0159/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0161/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0163/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0178/out_ohp.mtz \
  ../20221007_unscaled_unmerged/UCSF-P0179/out_ohp.mtz \
)

DW_LIST=None,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
USE_DW="DW"
# WILSON r ARRAY (if n+1 items, call sbatch with --array=0-n)
# r_string="scale=5; 1-0.4^${SLURM_ARRAY_TASK_ID}"
# R=$(bc -l <<< ${r_string} )
# RX=$R
DWR_LIST=0.,${R},${R},${R},${R},${R},${R},${R},${R},${R},${R},${R},${R},${R},${R},${R},${R},${R}


#Source your installation of careless here. To install careless, see: https://github.com/rs-station/careless
eval "$(conda shell.bash hook)"
conda activate careless


OUT=merge_${SLURM_JOB_ID}_${SEED}_${MODE}_mc1_10k_grid_${SLURM_ARRAY_TASK_ID}
#OUT=merge_${SLURM_JOB_ID}/r_${r}
mkdir -p $OUT
cp $0 $OUT
cat $0 > $OUT/slurm_script


SECONDS=0
CARELESS_ARGS=(
    --mc-samples=1
#    --learning-rate=0.001
    --separate-files
    --merge-half-datasets
    --half-dataset-repeats=$HALF_REPEATS
    --image-layers=$IL
    --dmin=$DMIN
    --iterations=$ITER
   --positional-encoding-frequencies=$PEF
   --positional-encoding-keys="XCAL,YCAL"
#    --wilson-prior-b 18.0
    --test-fraction=$TEST_FRACTION
    --seed=$SEED
    --mlp-width=25
)
# WARNING: bash [] comparisons only work for integers
c=$(echo "$R > -0.01" | bc)
if [ $c = '1' ]
then
  CARELESS_ARGS+=(--double-wilson-parents=${DW_LIST}) 
  CARELESS_ARGS+=(--double-wilson-r=${DWR_LIST})
fi
if [ $IL -lt 0 ]
then
  CARELESS_ARGS+=( --disable-image-scales)
fi
if [ $RU -gt 0 ]
then
  CARELESS_ARGS+=( --refine-uncertainties)
fi
if [ $STDOF -gt 0 ]
then
  CARELESS_ARGS+=( --studentt-likelihood-dof=$STDOF)
fi
CARELESS_ARGS+=( "Hobs,Kobs,Lobs,XCAL,YCAL,ZCAL,PSI,dHKL,ds_ref,ds_0,ds_1,ds_2,ds_3,ds_4,ds_5,ds_6,ds_7,ds_8,ds_9,ds_10,ds_11,ds_12,ds_13,ds_14,ds_15,ALF1,BET1,BATCH")

echo $CARELESS_ARGS
echo "Input MTZs: ${INPUT_MTZS[@]}" > ./$OUT/inputs_params.log
echo "Args: $MODE ${CARELESS_ARGS[@]}" >> ./$OUT/inputs_params.log
#echo "Careless version: ${CARELESS_VERSION}" >> ./$OUT/inputs_params.log
conda list > ./$OUT/conda_env_record.log

careless $MODE ${CARELESS_ARGS[@]} ${INPUT_MTZS[@]} $OUT/$BASENAME


# --- Cleanup --- #
echo "careless run complete."
#touch ./$OUT/inputs_params.log
mv myoutput*${SLURM_JOB_ID}* $OUT #DH added

DURATION=$SECONDS
TITLE="Slurm: careless"
MESSAGE="Job $SLURM_JOB_ID:careless finished on $HOSTNAME in $(($DURATION / 60)) minutes."
echo $MESSAGE
