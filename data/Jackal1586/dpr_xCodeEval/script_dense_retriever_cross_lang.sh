#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p PV1003q
#SBATCH -n 1
#SBATCH --nodelist=node14


# module add cuda11.2/toolkit/11.2.0
source /data/sbmaruf/anaconda3/bin/activate xcodeeval

export CUDA_VISIBLE_DEVICES='1,2,3,4,5,6,7'

for i in Javascript Go Perl Python Haskell Pascal C++ Scala D Rust C PHP Kotlin Ruby CS Java Ocaml; do
    python dense_retriever.py \
            model_file="/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/dumped_ret_xcodeeval/dpr_biencoder.39"  \
            qa_dataset=XCL_Retrieval_code_code_test \
            ctx_datatsets=[XCL_retrieval_$i] \
            encoded_ctx_files=[\"/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/emb_XCL_retrieval_"$i"_*\"] \
            out_file="/data/sbmaruf/zarzis/codebert_xCodeEval/16-39-51/XCL_Retrieval_code_code_test_ckpt39_$i.json"
done

# export -n CUDA_VISIBLE_DEVICES

# cd /export/home2/sbmaruf/prompt-tuning/prompt-tuning/
# sbatch scripts/train/t0_t5-3b/01.mem_prompt_2.sh



