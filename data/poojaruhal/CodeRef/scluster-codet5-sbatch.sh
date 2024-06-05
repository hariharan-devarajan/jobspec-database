#!/bin/bash
#SBATCH --job-name="Impact_Pretraining_CodeT5_Large"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=248G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 --constraint="GPUMEM80GB"
#SBATCH --time=09:30:00
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out

module purge all
module load multigpu
module load mamba

# Activate correct conda environment
#conda init bash
source ~/miniconda3/bin/activate
conda activate /home/ppooja/data/conda/envs/shirin-codet5

# install requirements when you create the environment first-time
#pip install -r requirements.txt

#login wandb when you create the environment the first time
wandb login <your-auth-key>

#chmod +x ./sh/pre-train.sh

srun ./sh/pre-train.sh
# Put your code below this line
#           $1: train_type, $2: train_mode, $3: model_name, $4: model_type, $5: train_languages, $6: test_languages,$7: jurisdiction, $8: data_augmentation_type, $9: train_sub_datasets ${10}: sub_datasets
#bash run.sh --train_type=$1 --train_mode=$2 --model_name=$3 --model_type=$4 --train_languages=$5 --test_languages=$6 --jurisdiction=$7 --data_augmentation_type=$8 --train_sub_datasets=$9 --sub_datasets=${10} --seed=${SLURM_ARRAY_TASK_ID} --debug=False >current-run.out
#srun python main.py -gm 32 -t mapa_coarse -lmt general_multilingual_all -ld NER_tasks_only --pty bash -1


# Example: bash run.sh --train_type=adapters --train_mode=train --model_name=xlm-roberta-base --model_type=hierarchical --train_languages=it --test_languages=it --jurisdiction=switzerland --data_augmentation_type=no_augmentation --train_sub_datasets=civil_law --sub_datasets=False --seed=1 --debug=True

# IMPORTANT:
# Run with                  sbatch run_sbatch.sh
# check with                squeue -u ppooja --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    sinfo and then nvidia-smi
# run interactive job with  srun ./sh/pre-train.sh --pty /bin/bash
