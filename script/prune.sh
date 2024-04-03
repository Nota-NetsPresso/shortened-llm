#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

run_command () {
    python src/block_prune.py \
     --base_model $1 --num_pruned_blocks $3 \
     --block_order_csv output_block_sensitivity/$2/ppl_n${4}/block_order.csv \
     --output_dir output_prune/$2/ppl_n$4/rm_${3}_blocks
}

# 20% / 27% / 35% pruning on LLaMA-7B
run_command "baffo32/decapoda-research-llama-7B-hf" "llama-1-7b" "6" "10"
run_command "baffo32/decapoda-research-llama-7B-hf" "llama-1-7b" "9" "10"
run_command "baffo32/decapoda-research-llama-7B-hf" "llama-1-7b" "11" "10"

# 20% / 27% / 35% pruning on Vicuna-7B-v1.3
run_command "lmsys/vicuna-7b-v1.3" "vicuna-7b-v1.3" "6" "10"
run_command "lmsys/vicuna-7b-v1.3" "vicuna-7b-v1.3" "9" "10"
run_command "lmsys/vicuna-7b-v1.3" "vicuna-7b-v1.3" "11" "10"

# 21% / 29% / 37% pruning on Vicuna-13B-v1.3
run_command "lmsys/vicuna-13b-v1.3" "vicuna-13b-v1.3" "8" "10"
run_command "lmsys/vicuna-13b-v1.3" "vicuna-13b-v1.3" "11" "10"
run_command "lmsys/vicuna-13b-v1.3" "vicuna-13b-v1.3" "15" "10"