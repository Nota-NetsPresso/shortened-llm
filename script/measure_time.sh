#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

run_command () {    
    for batch_size in {1,16,8,32,64,128,256}; do
        for max_seq_len in {128,512}; do
            python src/gen_batch_eval_time.py --base_model $1 \
                --output_dir results_efficiency/$2/batch_gen_out${max_seq_len}_bs${batch_size} \
                --batch_size $batch_size --max_seq_len $max_seq_len $3 
        done
    done
}

run_command "baffo32/decapoda-research-llama-7B-hf" "llama-1-7b" "--fix_decapoda_config"
run_command "nota-ai/st-llama-1-5.5b-ppl" "st-llama-1-5.5b-ppl" "--fix_decapoda_config"

run_command "lmsys/vicuna-13b-v1.3" "vicuna-13b-v1.3" ""
run_command "nota-ai/st-vicuna-v1.3-10.5b-ppl" "st-vicuna-v1.3-10.5b-ppl" ""