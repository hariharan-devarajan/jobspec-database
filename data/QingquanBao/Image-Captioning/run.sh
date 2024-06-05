#!/bin/bash

#SBATCH -p a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

#module load miniconda3
#__conda_setup="$('/dssg/opt/icelake/linux-centos8-icelake/gcc-11.2.0/miniconda3-4.10.3-f5dsmdmzng2ck6a4otduqwosi22kacfl/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#eval "$__conda_setup"
#conda activate pytorch


# training
python main.py train_evaluate --config_file configs/resnet101_attention_schedule.yaml 
# evaluate
python evaluate.py --prediction_file experiments/resnet101_attention_schedule2/resnet101_attention_b128_emd300_predictions.json \
                 --reference_file /dssg/home/acct-stu/stu464/data/image_caption/caption.txt \
                 --output_file result.txt