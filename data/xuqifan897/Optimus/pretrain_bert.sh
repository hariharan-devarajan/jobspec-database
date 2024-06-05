#!/bin/bash
#SBATCH -J emb
#SBATCH -o embo.txt
#SBATCH -e embe.txt
#SBATCH -p rtx
#SBATCH -N 2
#SBATCH -n 8
#SBATCH -t 00:30:00

module load cuda/10.1
source $HOME/programs/anaconda3/bin/activate
conda activate SUMMA

CHECKPOINT_PATH=checkpoint
DATA_PATH=/work2/07789/xuqifan/frontera/dataset/bert_data/my-bert_text_sentence
VOCAB_FILE=/work/07789/xuqifan/frontera/dataset/bert_data/vocab.txt

srun python pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --batch-size 64 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --checkpoint-activations \
       --distribute-checkpointed-activations \
       --train-iters 2000000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --tensorboard-dir $CHECKPOINT_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --min-lr 0.00001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 1 \
       --save-interval 1 \
       --eval-interval 1 \
       --eval-iters 1
