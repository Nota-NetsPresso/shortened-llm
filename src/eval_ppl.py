"""
Code modified from
https://github.com/horseee/LLM-Pruner/blob/main/LLMPruner/evaluator/ppl.py
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from dataset import get_loaders
from tqdm import tqdm
from utils import count_params, get_model, set_seed


@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss)
    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def eval_ppl(
    output_dir,
    model,
    tokenizer,
    datasets=["wikitext2", "ptb"],
    max_seq_len=128,
    batch_size=4,
    device="cuda",
    add_bos_to_every=False,
):
    filename = "ppl_bos.csv" if add_bos_to_every else "ppl.csv"
    csv_log_path = os.path.join(output_dir, filename)
    csv_header = []
    csv_value = []
    metric = {}
    for dataset in datasets:
        t0 = time.perf_counter()
        _, test_loader = get_loaders(
            dataset, tokenizer, max_seq_len, batch_size, add_bos_to_every
        )
        metric[dataset] = llama_eval(model, test_loader, device)

        print(
            f"PPL-{dataset}: {metric[dataset]} | add_bos_to_every: {add_bos_to_every} | time: {time.perf_counter()-t0:.1f}"
        )
        csv_header.append(f"ppl_{dataset}")
        csv_value.append(metric[dataset])

    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Current GPU memory occupied: {mem} MiB")
    nparams = count_params(model)
    print(f"Params: {nparams}")

    with open(csv_log_path, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(csv_header + ["params", "mem"])
        logwriter.writerow(csv_value + [nparams, mem])


def generate_txt(
    output_dir,
    model,
    tokenizer,
    input_prompt="The Leaning Tower of Pisa is known for",
    num_output=5,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    max_seq_len=128,
    device="cuda",
):
    # generate a few samples
    txt_path = os.path.join(output_dir, "gen_text.txt")
    inputs = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to(device)
    input_len = inputs[0].size(0)

    with open(txt_path, "w", encoding="utf8") as f:
        f.write("=== input ===\n")
        f.write(f"{input_prompt}\n")

    for i in range(num_output):
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_length=(input_len + max_seq_len),
                min_length=(
                    input_len + max_seq_len
                ),  # forced output length (to avoid <EOS> sampling)
                return_dict_in_generate=True,
            )
        s = generation_output.sequences[0]
        output_len = len(s)
        output = tokenizer.decode(s)

        print(f"=== output {i} | leng gen {output_len-input_len} + input {input_len}\n")
        print(output)

        with open(txt_path, "a", encoding="utf8") as f:
            f.write(
                f"=== output {i} | leng gen {output_len-input_len} + input {input_len}\n"
            )
            f.write(f"{output}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="baffo32/decapoda-research-llama-7B-hf",
        help="base model name",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="if None, base model name is used"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrain",
        choices=["pretrain", "pruneLLM", "tune_pruneLLM"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--input_prompt", type=str, default="The Leaning Tower of Pisa is known for"
    )
    parser.add_argument("--num_output", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/llama-7b-hf/ppl")
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer, description = get_model(
        base_model=args.base_model,
        ckpt=args.ckpt,
        lora_ckpt=args.lora_ckpt,
        tokenizer=args.tokenizer,
        model_type=args.model_type,
        device=args.device,
        fix_decapoda_config=args.fix_decapoda_config,
        use_bfloat=args.use_bfloat,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for add_bos_to_every in [True, False]:
        eval_ppl(
            output_dir=args.output_dir,
            model=model,
            tokenizer=tokenizer,
            datasets=["wikitext2", "ptb"],
            max_seq_len=args.max_seq_len,
            device=args.device,
            add_bos_to_every=add_bos_to_every,
        )
    generate_txt(
        output_dir=args.output_dir,
        model=model,
        tokenizer=tokenizer,
        input_prompt=args.input_prompt,
        num_output=args.num_output,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )
