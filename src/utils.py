import copy
import csv
import gc
import json
import random
import time

import numpy as np
import torch
from LLMPruner.peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def set_model_device_evalmode(
    model, device, fix_decapoda_config=False, use_bfloat=False
):
    if "cuda" in device:
        model.half()
        model = model.to(device)

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    model.eval()

    if use_bfloat:
        model = model.bfloat16()

    gc.collect()
    torch.cuda.empty_cache()

    return model


def get_model(
    base_model=None,
    ckpt=None,
    lora_ckpt=None,
    tokenizer=None,
    model_type="pretrain",
    device="cuda",
    fix_decapoda_config=False,
    use_bfloat=False,
):
    tokenizer = base_model if tokenizer is None else tokenizer
    if model_type == "pretrain":
        config = AutoConfig.from_pretrained(base_model)
        if "gptq" in base_model.lower():
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                base_model,
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=False,
                quantize_config=None,
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif (
            "LlamaForCausalLM" in config.__getattribute__("architectures")
            and "llama-3" not in base_model.lower()
        ):
            model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    elif model_type in ["pruneLLM", "tune_pruneLLM"]:
        pruned_dict = torch.load(ckpt, map_location="cpu")
        model = pruned_dict["model"]
        tokenizer = pruned_dict["tokenizer"]
        if model_type == "tune_pruneLLM":
            model = PeftModel.from_pretrained(
                model, lora_ckpt, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
    else:
        raise NotImplementedError
    description = "Model Type: {}\n Base: {} \n Pruned: {}\n LORA: {}".format(
        model_type, base_model, ckpt, lora_ckpt
    )

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        tokenizer.pad_token_id = 0
    model = set_model_device_evalmode(model, device, fix_decapoda_config, use_bfloat)

    return model, tokenizer, description


def copy_weight(model, model_orig, list_pruned_blocks):
    connect_info = {}  # connect_info['TO-small'] = 'FROM-orig'
    connect_info["model.embed_tokens.weight"] = "model.embed_tokens.weight"
    connect_info["model.norm.weight"] = "model.norm.weight"
    connect_info["lm_head.weight"] = "lm_head.weight"

    k = 0
    for k_orig in range(model_orig.config.__getattribute__("num_hidden_layers")):
        if k_orig in list_pruned_blocks:  # uncopied = pruned blocks
            continue
        connect_info[f"model.layers.{k}."] = f"model.layers.{k_orig}."
        print(f"original model.layers.{k_orig} --> pruned model.layers.{k}")
        k = k + 1

    print(f" ** excluded blocks {list_pruned_blocks}")

    t0 = time.perf_counter()
    for k in model.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])
                break
        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
            model.state_dict()[k].copy_(model_orig.state_dict()[k_orig])
    print(f"copy time --- {(time.perf_counter()-t0):.1f} sec")

    return model


def get_block_pruned_network(
    model_orig,
    unimportance_order,
    num_pruned_blocks=1,
    device="cuda",
    fix_decapoda_config=False,
    use_bfloat=False,
):
    # Define the block-pruned architecture with random initialization
    config = copy.deepcopy(model_orig.config)
    print(f"# blocks before pruning: {config.num_hidden_layers}")
    config.__setattr__(
        "num_hidden_layers", (config.num_hidden_layers - num_pruned_blocks)
    )
    print(f"# blocks after pruning: {config.num_hidden_layers}")
    model_pruned = AutoModelForCausalLM.from_config(config)

    # Copy the original model's weights to the pruned model
    model_pruned = copy_weight(
        model_pruned, model_orig, unimportance_order[:num_pruned_blocks]
    )
    model_pruned = set_model_device_evalmode(
        model_pruned, device, fix_decapoda_config, use_bfloat
    )
    return model_pruned


def convert_json2csv_zeroshot(json_path, csv_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    select_key = {
        "boolq": "acc",
        "piqa": "acc",
        "hellaswag": "acc_norm",
        "winogrande": "acc",
        "arc_easy": "acc",
        "arc_challenge": "acc_norm",
        "openbookqa": "acc_norm",
    }

    list_task = []
    list_metric = []
    list_score = []

    ave_score = 0
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for name, key in select_key.items():
            list_task.append(name)
            list_metric.append(key)

            score = data["results"][name][key] * 100
            list_score.append(score)
            ave_score += score

        ave_score /= len(select_key)

        list_task.append("AVE")
        list_metric.append("n/a")
        list_score.append(ave_score)

        writer.writerow(list_task)
        writer.writerow(list_metric)
        writer.writerow(list_score)

    print(csv_path)
