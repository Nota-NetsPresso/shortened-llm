import argparse
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer
from LLMPruner.peft import PeftModel
import time
import gc
import json
import csv
import copy

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

def set_model_device_evalmode(model, device, fix_decapoda_config=True):
    if "cuda" in device:
        model.half()
        model = model.to(device)

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    model.eval()    

    gc.collect()
    torch.cuda.empty_cache()    
    
    return model

def get_model(base_model=None, ckpt=None, lora_ckpt=None, tokenizer=None,
              model_type='pretrain', device='cuda', fix_decapoda_config=True):    
    tokenizer = base_model if tokenizer is None else tokenizer
    if model_type == 'pretrain':       
        config = AutoConfig.from_pretrained(base_model)
        if "LlamaForCausalLM" in config.__getattribute__("architectures"): 
            model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)       
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
        else:
            model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)       
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    elif model_type in ['pruneLLM', 'tune_pruneLLM']:
        pruned_dict = torch.load(ckpt, map_location='cpu')        
        model = pruned_dict['model']
        tokenizer = pruned_dict['tokenizer']
        if model_type == 'tune_pruneLLM':
            model = PeftModel.from_pretrained(model, lora_ckpt, torch_dtype=torch.float16, low_cpu_mem_usage=True)       
    else:
        raise NotImplementedError
    description = "Model Type: {}\n Base: {} \n Pruned: {}\n LORA: {}".format(model_type, base_model, ckpt, lora_ckpt)

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        tokenizer.pad_token_id = 0
    model = set_model_device_evalmode(model, device, fix_decapoda_config)

    return model, tokenizer, description

def convert_json2csv_zeroshot(json_path, csv_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    select_key = {
        'boolq': 'acc',
        'piqa': 'acc',
        'hellaswag': 'acc_norm',
        'winogrande': 'acc',
        'arc_easy': 'acc',
        'arc_challenge': 'acc_norm',
        'openbookqa': 'acc_norm',
    }

    list_task = []
    list_metric = []
    list_score = []

    ave_score = 0
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)       
        for name, key in select_key.items():
            list_task.append(name)
            list_metric.append(key)
            
            score = data['results'][name][key]*100
            list_score.append(score)            
            ave_score += score
        
        ave_score /= len(select_key)

        list_task.append('AVE')
        list_metric.append('n/a')
        list_score.append(ave_score)

        writer.writerow(list_task)
        writer.writerow(list_metric)
        writer.writerow(list_score)

    print(csv_path)    