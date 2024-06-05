#!/bin/bash
#SBATCH --job-name=example
#SBATCH --account=Project_2001426
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail

#module load gcc/8.3.0 cuda/10.1.168

#rm test.log

module purge
#module load python-data
module load tensorflow/2.2-hvd
#module load gcc/8.3.0 cuda/10.1.168
source tf2.2-transformers3.4/bin/activate
#source venv/bin/activate
#source venv_transformers_ner/bin/activate
#python3 scripts/preprocess.py data/dev.txt.tmp $BERT_MODEL $MAX_LENGTH  > data/dev.txt

#
# default values
batch_size=16
learning_rate="4e-5"
epochs=4
max_seq_length=256
model="./models/monologg/biobert_v1.1_pubmed/"
#output_dir="./models/s800_1"
grid="false"

for i in "$@"
do
case $i in
    -g=*|--grid=*)
    grid="${i#*=}"
    shift
    ;;
    -bs=*|--batch_size=*)
    batch_size="${i#*=}"
    shift # past argument=value
    ;;
    -lr=*|--learning_rate=*)
    learning_rate="${i#*=}"
    shift # past argument=value
    ;;
    -e=*|--epochs=*)
    epochs="${i#*=}"
    shift # past argument=value
    ;;
    -msl=*|--max_seq_length=*)
    max_seq_length="${i#*=}"
    shift # past argument=value
    ;;
    -m=*|--model=*)
    model="${i#*=}"
    shift # past argument=value
    ;;
    -lr=*|--learning_rate=*)
    learning_rate="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--data_dir=*)
    data_dir="${i#*=}"
    shift # past argument=value
    ;;
    -od=*|--output_dir=*)
    output_dir="${i#*=}"
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done

if [ $grid == "false" ]; then
  rm -f latest.out latest.err
  ln -s logs/$SLURM_JOBID.out latest.out
  ln -s logs/$SLURM_JOBID.err latest.err
fi

#MAX_LENGTH=128
#BERT_MODEL=bert-base-multilingual-cased
#BERT_MODEL="monologg/biobert_v1.0_pubmed_pmc"
#BERT_MODEL="monologg/biobert_v1.1_pubmed"
#BERT_MODEL="./models/biobertTorch"
#model="./models/biobertTorch"

#BERT_MODEL="./biobert/biobert_v1.0_pubmed_pmc"
#OUTPUT_DIR="./models/s800_1"
#BATCH_SIZE=32
#NUM_EPOCHS=3
SAVE_STEPS=0
SEED=1
#DATADIR="./data/s800/conll/" #s800SmallTrain/" #/conll/"
#LABELS="${DATADIR}labels.txt"
#LEARNING_RATE="0.001"

#rm -rf $OUTPUT_DIR

labels=$data_dir"labels.txt"
echo "Using the following parameters"
echo "#############################"
echo "batch size: $batch_size"
echo "maximum sequence length: $max_seq_length"
echo "epochs: $epochs"
echo "data: $data_dir"
echo "labels: $labels"
echo "model: $model"
echo "learning rate: $learning_rate"
echo "output dir: $output_dir"
echo "savesteps: $SAVE_STEPS"
echo "seed: $SEED"

#echo "Sample from conll:"
#head -5 $data_dir"train.txt"
#echo "Sample from labels:"
#head -5 $labels

if [ -f test.log ]; then
 rm test.log
fi

if [ -f test_utils_ner.log ]; then
 rm test_utils_ner.log
fi

if [ ! -f "$labels" ]; then
  echo "$labels not found"
  exit 1
fi


echo "Running python program"
python3 run_tf_ner.py --data_dir $data_dir \
--labels $labels \
--model_name_or_path $model \
--output_dir $output_dir \
--max_seq_length  $max_seq_length \
--num_train_epochs $epochs \
--per_device_train_batch_size $batch_size \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir \
--learning_rate $learning_rate \

#--from_pt True\


