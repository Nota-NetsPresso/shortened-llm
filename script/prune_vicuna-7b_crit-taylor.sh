#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL=lmsys/vicuna-7b-v1.3
export MODEL_NAME=vicuna-7b-v1.3
export NUM_CALIB_DATA=10
export NUM_PRUNED_BLOCKS=6
export OUTPUT_SENSITIVITY=output_block_sensitivity/$MODEL_NAME/taylor_n${NUM_CALIB_DATA}
export OUTPUT_PRUNE=output_prune/$MODEL_NAME/taylor_n${NUM_CALIB_DATA}/rm_${NUM_PRUNED_BLOCKS}_blocks
export OUTPUT_TUNE=output_tune/$MODEL_NAME/taylor_n${NUM_CALIB_DATA}/rm_${NUM_PRUNED_BLOCKS}_blocks

# Analyze the taylor-based block importance with 10 calibration samples
python src/anal_block_sensitivity_taylor.py \
    --base_model $BASE_MODEL \
    --num_calib_data $NUM_CALIB_DATA \
    --output_dir $OUTPUT_SENSITIVITY --batch_size 1

# Perform 20% block pruning by removing 6 Transformer Blocks
python src/block_prune.py \
    --base_model $BASE_MODEL \
    --num_pruned_blocks $NUM_PRUNED_BLOCKS \
    --block_order_csv $OUTPUT_SENSITIVITY/block_order.csv \
    --output_dir $OUTPUT_PRUNE

# Perform LoRA-based retraining
python src/lora_retrain.py \
    --base_model $OUTPUT_PRUNE \
    --data_path yahma/alpaca-cleaned \
    --output_dir $OUTPUT_TUNE \
    --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --save_lora_merge

# Compute Zero-shot PPL on WikiText2 and PTB 
python src/eval_ppl.py \
    --base_model ${OUTPUT_TUNE}_lora_merge_fp16 \
    --output_dir ${OUTPUT_TUNE}_score

# Compute Zero-shot accuracy on seven commonsense reasoning tasks 
python src/eval_zeroshot_acc.py \
    --model hf-causal-experimental --no_cache \
    --model_args pretrained=${OUTPUT_TUNE}_lora_merge_fp16 \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda \
    --output_json ${OUTPUT_TUNE}_score/zeroshot_acc.json \
    | tee ${OUTPUT_TUNE}_score/zeroshot_acc.txt