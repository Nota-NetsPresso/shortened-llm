import argparse
import logging
import random

from transformers import AutoTokenizer
import torch
import numpy as np
from datasets import load_dataset
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from utils import set_seed
from dataset import get_examples

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_dataset(nsamples, seed, seqlen, tokenizer, buffer_size=5000):
    # set random seed for reproducibility
    set_seed(seed)

    # load raw dataset from Huggingface datasets
    raw_data = get_examples(
        dataset="c4",
        tokenizer=tokenizer,
        n_samples=nsamples,
        return_raw_dataset=True,
    )

    # concaenate long enough text
    all_text = "\n\n".join(
        raw_data["text"][:buffer_size]
    )  # further reduce sample size (large enough to cover nsamples*seqlen tokens)

    encoding = tokenizer(all_text, return_tensors="pt")

    # gather dataset
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, encoding.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoding.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


def quantize(base_model, tokenizer_name=None, quantized_model_dir=None):
    if tokenizer_name is None:
        tokenizer_name = base_model

    if quantized_model_dir is None:
        quantized_model_dir = f"{base_model.split('/')[-1]}-GPTQ"

    logging.info(f"base_model = {base_model}, tokenizer = {tokenizer_name}")

    # base quantization config example # TODO. choose optimal one
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )  # ppl is important than latency for now

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(base_model, quantize_config)

    # get calibration data
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    examples = get_dataset(
        nsamples=128,
        seed=0,
        seqlen=2048,
        tokenizer=tokenizer,
    )

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    print(
        'quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"'
    )
    model.quantize(examples, use_triton=False)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)
    tokenizer.save_pretrained(quantized_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="",
        help="base model name",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="if None, base model name is used"
    )
    parser.add_argument(
        "--quantized_model_dir",
        type=str,
        default=None,
        help="if None, it is inferred from base_model",
    )

    args = parser.parse_args()

    model = quantize(
        base_model=args.base_model,
        tokenizer_name=args.tokenizer,
        quantized_model_dir=args.quantized_model_dir,
    )
