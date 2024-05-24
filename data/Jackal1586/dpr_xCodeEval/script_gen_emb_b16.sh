#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p RTX8Kq
#SBATCH -n 1
#SBATCH --nodelist=node22

module add cuda11.2/toolkit/11.2.0
source /home/sbmaruf/anaconda3/bin/activate xcodeeval

export CUDA_VISIBLE_DEVICES='0,1,2'


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Javascript     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Javascript"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Go     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Go"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Perl     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Perl"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Python     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Python"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Haskell     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Haskell"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Pascal     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Pascal"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_C++     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_C++"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Scala     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Scala"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_D     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_D"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Rust     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Rust"


python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_C     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_C"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_PHP     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_PHP"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Kotlin     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Kotlin"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Ruby     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Ruby"


python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_CS     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_CS"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Java     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Java"


#python generate_dense_embeddings.py     model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/dumped_ret_xcodeeval/dpr_biencoder.35"     ctx_src=XCL_retrieval_Ocaml     shard_id=0 num_shards=1     out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-28/emb_XCL_retrieval_Ocaml"

export -n CUDA_VISIBLE_DEVICES

cd /export/home2/sbmaruf/prompt-tuning/prompt-tuning/
sbatch scripts/train/t0_t5-3b/01.mem_prompt_2.sh


   
