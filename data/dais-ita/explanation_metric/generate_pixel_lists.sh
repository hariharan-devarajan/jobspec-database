#!/bin/sh --login

#SCW_TPN_OVERRIDE=1 #overide warning about 1 node
# BASH Environment Variable	           SBATCH Field Code	Description
# $SLURM_JOB_ID	                        %J	                Job identifier
# $SLURM_ARRAY_JOB_ID	                %A	                Array parent job identifier
# $SLURM_ARRAY_TASK_ID	                %a	                Array job iteration index

#SBATCH --gres=gpu:1
#SBATCH -p gpu



experiment_id="$1"
echo experiment_id: "$experiment_id"

data_range_index_start="$2"
echo data_range_index_start: "$data_range_index_start"

data_range_index_end="$3"
echo data_range_index_end: "$data_range_index_end"

explanation_name="$4"
echo explanation_name: "$explanation_name"

generate_for_test_data="$5"
echo generate_for_test_data: "$generate_for_test_data"

dataset_name="$6"
echo dataset_name: "$dataset_name"

model_name="$7"
echo model_name: "$model_name"


#SBATCH --job-name="generate_pixel_list_${experiment_id}_${data_range_index_start}_${data_range_index_end}"
#SBATCH -o output-%J.o
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

module purge
module load compiler/intel/2018 tensorflow/1.11
source /home/c.c0919382/envs/metric_env/bin/activate
python arcca_generate_pixel_lists.py $experiment_id $data_range_index_start $data_range_index_end $explanation_name $generate_for_test_data $dataset_name $model_name