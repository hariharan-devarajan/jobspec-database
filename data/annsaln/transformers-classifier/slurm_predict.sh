#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -t 01:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2002026
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

echo "START: $(date)"

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module load tensorflow/2.2
#list-packages
#python3 -m venv test
#source test/bin/activate
#export PYTHONPATH=test/lib/python3.6/site-packages:$PYTHONPATH
#pip3 install --upgrade pip
#pip install transformers==3.4
#pip3 install scikit-learn
#pip install tensorflow-addons==0.12
#pip install transformers sentencepiece
#2.4-hvd
#module load python-data
#source transformers3.4tf2.4/bin/activate

#export PYTHONPATH=/scratch/project_2002026/multilabel_bert/svregisters/lstm/transformer-classifier/transformers3.4tf3.4/lib/python3.7/site-packages:$PYTHONPATH

#MODEL=$1
#MODEL="jplu/tf-xlm-roberta-large"
MODEL="jplu/tf-xlm-roberta-base"
#MODEL_ALIAS=$2
#SRC=$3
#TRG=$4
#LR_=7e-5
EPOCHS=0
#i=1
BS=128
PRED=$1

#echo "MODEL:$MODEL"
#echo "MODEL_ALIAS:$MODEL_ALIAS"
#echo "SRC:$SRC"
#echo "TRG:$TRG"
#echo "LR:$LR_"
#echo "EPOCHS:$EPOCHS_"
#echo "i:$i"

#export TRAIN_DIR=junkdata/$SRC
#export DEV_DIR=junkdata/$TRG
export OUTPUT_DIR=models
export PRED_DIR=predictions

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PRED_DIR"

#for EPOCHS in $EPOCHS_; do
#for LR in $LR_; do
#for j in $i; do
#echo "Settings: src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS"
#echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/experiments.log
srun python train_lstm.py \
  --model_name $MODEL \
  --pred $PRED \
  --input_format json \
  --seq_len 512 \
  --max_lines 50 \
  --epochs $EPOCHS \
  --batch_size $BS \
  --load_model $2 \
  --lstm_model $3 \
  --threshold 0.5 \
  --save_predictions $PRED_DIR/$PRED.classified \
#  --log_file logs/train_$MODEL_ALIAS-$SRC-$TRG.tsv \
#  --test_log_file logs/test_$MODEL_ALIAS-$SRC-$TRG.tsv \
#  --output_file $OUTPUT_DIR/$MODEL_ALIAS-$SRC-$TRG-lr$LR-ep$EPOCHS-$j.h5 
#  --train $TRAIN_DIR/train.json \
#  --dev $TRAIN_DIR/dev.json \
#  --test $DEV_DIR/test.json \

# SWEDISH MODELS:
#xmlr-sv-sv-lr2e-6-ep5-1.h5
#lstm50-sv-sv-lr7e-5-ep200-2.h5
#FRENCH:
#xmlr-fr-fr-lr2e-6-ep6-1.h5
#lstm-fr-fr-lr7e-5-ep200-1.h5
#MULTILING:
#xmlr-de-en-es-fi-fr-se-de-en-es-fi-fr-se-lr2e-6-ep6-1.h5
#lstm-de-en-es-fi-fr-se-de-en-es-fi-fr-se-lr4e-5-ep200-1.h5


#  --multiclass
#--output_file $OUTPUT_DIR/model.h5 \
#  --load_model $OUTPUT_DIR/model_nblocks3-ep10-2.h5 \
#--load_model $OUTPUT_DIR/model.h5 \
# swedish model xmlr-sv-sv-lr2e-6-ep5-1.h5 f-score 80
# french model xmlr-fr-fr-lr8e-6-ep3-1.h5
# xmlr-de-en-es-fi-fr-sv-lr2e-6-ep6-1.h5
# xmlr-de-en-es-fi-se-fr-lr2e-6-ep6-1.h5 
# xmlr-de-es-fi-fr-se-en-lr2e-6-ep5-5.h5
# xmlr-$SRC-$TRG-lr2e-6-ep6-1.h5
#done
#done
#done

echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/completed.log


#rm -rf "$OUTPUT_DIR"
#mkdir -p "$OUTPUT_DIR"


echo "END: $(date)"
