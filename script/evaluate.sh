#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

run_command () {
    python src/eval_ppl.py --base_model $1 --output_dir results/$2/ppl $3

    python src/eval_zeroshot_acc.py \
        --model hf-causal-experimental --no_cache \
        --model_args pretrained=$1 \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --device cuda --output_json results/$2/zeroshot_acc.json | tee results/$2/zeroshot_acc.txt
}

# Shortened LLMs available at HuggingFace Hub
run_command "nota-ai/st-llama-1-5.5b-taylor" "st-llama-1-5.5b-taylor" "--fix_decapoda_config"
run_command "nota-ai/st-llama-1-5.5b-ppl" "st-llama-1-5.5b-ppl" "--fix_decapoda_config"
run_command "nota-ai/st-vicuna-v1.3-5.5b-ppl" "st-vicuna-v1.3-5.5b-ppl" ""
run_command "nota-ai/st-vicuna-v1.3-5.5b-taylor" "st-vicuna-v1.3-5.5b-taylor" ""
run_command "nota-ai/st-vicuna-v1.3-10.5b-ppl" "st-vicuna-v1.3-10.5b-ppl" ""
run_command "nota-ai/st-vicuna-v1.3-10.5b-taylor" "st-vicuna-v1.3-10.5b-taylor" ""

# Original Models
run_command "baffo32/decapoda-research-llama-7B-hf" "llama-1-7b" "--fix_decapoda_config"
run_command "lmsys/vicuna-7b-v1.3" "vicuna-7b-v1.3" ""
run_command "lmsys/vicuna-13b-v1.3" "vicuna-13b-v1.3" ""
run_command "rishiraj/CatPPT-base" "CatPPT-base" ""
run_command "google/gemma-2b" "gemma-2b" ""
run_command "google/gemma-7b" "gemma-7b" ""
run_command "meta-llama/Meta-Llama-3-8B" "Meta-Llama-3-8B" ""
run_command "meta-llama/Llama-2-7b-hf" "Llama-2-7b-hf" "--use_bfloat"