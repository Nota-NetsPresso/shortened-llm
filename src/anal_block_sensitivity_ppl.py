import argparse
import csv
import os
import time

import numpy as np
import torch
from dataset import get_examples
from utils import count_params, get_block_pruned_network, get_model, set_seed


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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--num_calib_data", type=int, default=10)
    parser.add_argument(
        "--output_dir", type=str, default="output_block_sensitivity/llama-1-7b/ppl_n10"
    )
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument(
        "--add_bos_to_every",
        default=False,
        action="store_true",
        help="whether to add BOS token to every sample in calibration dataset",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    unsorted_output_path = os.path.join(args.output_dir, "all_ppl_unsorted.csv")
    sorted_output_path = os.path.join(args.output_dir, "all_ppl_sorted.csv")
    block_order_path = os.path.join(args.output_dir, "block_order.csv")

    if not os.path.exists(block_order_path):
        model_orig, tokenizer, description = get_model(
            base_model=args.base_model,
            ckpt=args.ckpt,
            lora_ckpt=args.lora_ckpt,
            tokenizer=args.tokenizer,
            model_type=args.model_type,
            device="cpu",
            fix_decapoda_config=args.fix_decapoda_config,
            use_bfloat=args.use_bfloat,
        )
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        # Evaluate the model with a single block removal
        example_prompts = get_examples(
            dataset="bookcorpus",
            tokenizer=tokenizer,
            n_samples=args.num_calib_data,
            seq_len=args.max_seq_len,
            field_name="text",
            add_bos_to_every=args.add_bos_to_every,
        ).to(args.device)

        for block_idx in range(model_orig.config.__getattribute__("num_hidden_layers")):
            csv_log_path = os.path.join(
                args.output_dir, f"ppl_block{block_idx}_removed.csv"
            )
            if os.path.exists(csv_log_path):
                print(f"already computed - {csv_log_path}")
                continue

            model = get_block_pruned_network(
                model_orig,
                unimportance_order=[block_idx],
                num_pruned_blocks=1,
                device=args.device,
            )

            # Measure PPL
            t0 = time.perf_counter()
            nlls = []
            with torch.no_grad():
                for j in range(args.num_calib_data):
                    x = example_prompts[j].unsqueeze(0)
                    output = model(x)
                    lm_logits = output.logits
                    shift_logits = lm_logits[:, :-1, :].contiguous()
                    shift_labels = x[:, 1:].contiguous()
                    loss = loss_fct(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    nlls.append(loss)

            ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())

            # Save
            with open(csv_log_path, "w") as logfile:
                logwriter = csv.writer(logfile, delimiter=",")
                logwriter.writerow(
                    ["removed_block", "ppl_bookcorpus", "num_calib_data", "params"]
                )
                logwriter.writerow(
                    [block_idx, ppl, args.num_calib_data, count_params(model)]
                )

            print(f"PPL over Bookcorpus {args.num_calib_data} samples: {ppl}")
            print(f"  * time in sec: {time.perf_counter()-t0}")
            del model

        # Collec the PPL-based block sensitivity results
        unsorted_results = []
        with open(unsorted_output_path, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(
                ["removed_block", "ppl_bookcorpus", "num_calib_data", "params"]
            )
            for block_idx in range(
                model_orig.config.__getattribute__("num_hidden_layers")
            ):
                csv_log_path = os.path.join(
                    args.output_dir, f"ppl_block{block_idx}_removed.csv"
                )
                with open(csv_log_path, "r") as file:
                    next(file)  # pass the header line
                    data = [float(i) for i in str(next(file).strip()).split(",")]
                    logwriter.writerow(data)
                    unsorted_results.append(data)

        sorted_results = sorted(unsorted_results, key=lambda x: x[1], reverse=False)

        block_order = []
        with open(sorted_output_path, "w") as logfile, open(
            block_order_path, "w"
        ) as logfile_order:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(
                ["removed_block", "ppl_bookcorpus", "num_calib_data", "params"]
            )
            logwriter.writerows(sorted_results)
            for data in sorted_results:
                block_order.append(int(data[0]))
            logwriter_order = csv.writer(logfile_order, delimiter=",")
            logwriter_order.writerow(block_order)

        print(f"=== block order removed: {block_order_path}")
        print(block_order)
        print(f"len: {len(block_order)}")

    else:
        print(f"use the precomputed results at {block_order_path}")
