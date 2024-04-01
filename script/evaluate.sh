#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

run_command () {
    python src/eval_ppl.py --base_model $1 --output_dir results/$2/ppl

    python src/eval_zeroshot_acc.py \
        --model hf-causal-experimental --no_cache \
        --model_args pretrained=$1 \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --device cuda --output_json results/$2/zeroshot_acc.json | tee results/$2/zeroshot_acc.txt
}

run_command "nota-ai/st-llama-1-5.5b-taylor" "st-llama-1-5.5b-taylor"
run_command "nota-ai/st-llama-1-5.5b-ppl" "st-llama-1-5.5b-ppl"
run_command "nota-ai/st-vicuna-v1.3-5.5b-ppl" "st-vicuna-v1.3-5.5b-ppl"
run_command "nota-ai/st-vicuna-v1.3-5.5b-taylor" "st-vicuna-v1.3-5.5b-taylor"
run_command "nota-ai/st-vicuna-v1.3-10.5b-ppl" "st-vicuna-v1.3-10.5b-ppl"
run_command "nota-ai/st-vicuna-v1.3-10.5b-taylor" "st-vicuna-v1.3-10.5b-taylor"