#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH --mem 30G # 20GB solicitados.
#SBATCH -p mhigh,mlow # or mlow Partition to submit to
#SBATCH --gres gpu:1 # Para pedir Pascales MAX 8
#SBATCH -o ./out/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e ./out/%x_%u_%j.err # File to which STDERR will be written

#SBATCH --job-name=udacity_new_xml_test_train

nvidia-smi
export CUDA_LAUNCH_BLOCKING=1

LEARNING_RATE=1e-3
BATCH_SIZE=1
DECAY_STEP=5

python3 test_net.py --dataset pascal_voc --net res101 \
                       --cuda --mGPUs --checksession 1 --checkepoch 20 --checkpoint 2504
