#!/bin/bash

evaluate() {
    python src/eval_ppl.py --base_model $1 --output_dir results/$2/ppl --device cuda $3

    python src/eval_zeroshot_acc.py \
        --model hf-causal-experimental --no_cache \
        --model_args pretrained=$1 \
        --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
        --device cuda --output_json results/$2/zeroshot_acc.json | tee results/$2/zeroshot_acc.txt
}

BASE_MODEL_PATHS=(
    "nota-ai/cpt_st-vicuna-v1.3-5.5b-ppl"
    "nota-ai/cpt_st-vicuna-v1.3-3.7b-ppl"
    "nota-ai/cpt_st-vicuna-v1.3-2.7b-ppl"
    "nota-ai/cpt_st-vicuna-v1.3-1.5b-ppl"

)

QUANTIZED_MODEL_DIRS=(
    "quantized_models/GPTQ/st-vcn-5.5b-CPT"
    "quantized_models/GPTQ/st-vcn-3.7b-CPT"
    "quantized_models/GPTQ/st-vcn-2.7b-CPT"
    "quantized_models/GPTQ/st-vcn-1.5b-CPT"
)

EVAL_NAMES=(
    "st-vcn-5.5b-CPT-GPTQ"
    "st-vcn-3.7b-CPT-GPTQ"
    "st-vcn-2.7b-CPT-GPTQ"
    "st-vcn-1.5b-CPT-GPTQ"
)

NUM_EVAL=${#BASE_MODEL_PATHS[@]}

for ((i = 0; i < $NUM_EVAL; i++)); do
    BASE_MODEL_PATH=${BASE_MODEL_PATHS[$i]}
    QUANTIZED_MODEL_DIR=${QUANTIZED_MODEL_DIRS[$i]}
    EVAL_NAME=${EVAL_NAMES[$i]}

    echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
    echo "QUANTIZED_MODEL_DIR: $QUANTIZED_MODEL_DIR"
    echo "EVAL_NAME: $EVAL_NAME"

    if [ ! -d "$QUANTIZED_MODEL_DIR" ]; then
        mkdir -p "$QUANTIZED_MODEL_DIR"
    fi

    # Run quantization
    python src/quantize_gptq.py --base_model $BASE_MODEL_PATH --quantized_model_dir $QUANTIZED_MODEL_DIR

    # Evaluate
    evaluate $QUANTIZED_MODEL_DIR $EVAL_NAME ""
done
