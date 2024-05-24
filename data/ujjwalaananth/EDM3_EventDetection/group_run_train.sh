#!/bin/bash
#SBATCH -n 20                        # number of cores
#SBATCH -t 01-00:00                  # wall time (D-HH:MM)
#SBATCH -N 1
##SBATCH -A uananthe
#SBATCH -p cidsegpu1
##SBATCH -p rcgpu7
#SBATCH -q cidsegpu1_contrib_res
##SBATCH -q wildfire
#SBATCH --gres=gpu:4
#SBATCH --mail-type=END,FAIL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=uananthe@asu.edu # send-to address
#SBATCH -o /scratch/uananthe/ADD_multi4_prtag_evex.%j.out             # STDOUT (%j = JobId)
#SBATCH -e /scratch/uananthe/ADD_multi4_prtag_evex.%j.err             # STDERR (%j = JobId)

#export TRANSFORMERS_CACHE=/scratch/mparmar3/.cache/                                                           
#export PYTHONPATH=$PYTHONPATH:/home/mparmar3/.conda/envs/mihir_pytorch/lib/python3.7/site-packages/
#data_path="/scratch/mparmar3/artefacts/data"
#data_type='20_data'

module purge;
module load anaconda3/5.3.0;
source activate secenv;

#declare -a dataset_list=("Quoref")

python model2/run_model_seq.py \
        --model_name_or_path '/scratch/uananthe/thesis/multitask4mixbiverbose2_prin_maven_1024_20_1bs_2DAYS' \
        --do_train false \
        --do_predict true \
        --train_file "/scratch/uananthe/EventEx_data/multitask4mix_bi_verbose2/trainmaven.csv" \
        --test_file "/scratch/uananthe/EventEx_data/multitask4mix_bi_verbose2/testmaven.csv" \
        --output_dir "/scratch/uananthe/thesis/multitask4mixbiverbose2_prin_maven_1024_20_bs50_seq3_1bs_2DAYS" \
        --overwrite_output_dir true \
        --num_train_epochs="100" \
        --max_source_length="1024" \
        --text_column="text_in" \
        --summary_column="src_label" \
        --per_device_train_batch_size="1" \
        --per_device_eval_batch_size="1" \
        --predict_with_generate \
        --save_strategy="steps" \
        --num_beams="50" \
        --num_return_sequences="3" #\
#        --num_beam_groups="5"
