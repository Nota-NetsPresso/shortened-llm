import argparse
import csv
import os

import torch
from dataset import get_examples
from utils import get_model, set_seed


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
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--num_calib_data", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_block_sensitivity/llama-1-7b/taylor_n10",
    )
    parser.add_argument("--norm_power", type=int, default=1, help="1 or 2 for l-p norm")
    parser.add_argument(
        "--weight_reduction", type=str, default="sum", help="sum, mean, max, prod"
    )
    parser.add_argument(
        "--block_reduction", type=str, default="sum", help="sum, mean, max, prod"
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

    norm_power = args.norm_power
    weight_reduction = args.weight_reduction
    block_reduction = args.block_reduction
    result_csv_weight = os.path.join(args.output_dir, "weight_score.csv")
    result_csv_block = os.path.join(args.output_dir, "block_score_all.csv")
    result_csv_block_detail = os.path.join(args.output_dir, "block_score_detail.csv")
    result_csv_block_sort = os.path.join(args.output_dir, "block_score_sorted.csv")
    block_order_path = os.path.join(args.output_dir, "block_order.csv")

    if not os.path.exists(block_order_path):
        # Do forward to collect gradient information
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
        example_prompts = get_examples(
            dataset="bookcorpus",
            tokenizer=tokenizer,
            n_samples=args.num_calib_data,
            seq_len=args.max_seq_len,
            field_name="text",
            add_bos_to_every=args.add_bos_to_every,
        ).to(args.device)

        print("Do forward to collect gradient information")
        salience_dict = {}
        for i in range(0, example_prompts.size(0), args.batch_size):
            example_prompts_tmp = example_prompts[i : i + args.batch_size]
            loss = model(example_prompts_tmp, labels=example_prompts_tmp).loss
            loss.backward()
            for k, param in model.named_parameters():
                if param.requires_grad and "weight" in k and "embed_tokens" not in k:
                    salience = param * param.grad
                    salience = salience.data.clone().float()

                    if k not in salience_dict.keys():
                        salience_dict[k] = salience
                    else:
                        salience_dict[k] += salience
            model.zero_grad()

        # Compute scores of weight matrices -> Collec them
        block_info = {}
        with open(result_csv_weight, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["weight_name", "weight_score"])
            for k, param in model.named_parameters():
                if param.requires_grad and "weight" in k and "embed_tokens" not in k:
                    block_idx = ".".join(k.split(".")[:3])  # 'model.layers.i'
                    if "proj" in k or "lm_head" in k:  # output_dim x input_dim
                        weight_imp = (
                            salience_dict[k].abs().pow(norm_power).sum(1)
                        )  # [output_dim]
                    elif "norm" in k:  # [output_dim]
                        weight_imp = salience_dict[k].abs().pow(norm_power)

                    if weight_reduction == "sum":
                        weight_imp = weight_imp.sum(dim=0)
                    elif weight_reduction == "mean":
                        weight_imp = weight_imp.mean(dim=0)
                    elif weight_reduction == "max":
                        weight_imp = weight_imp.max(dim=0)[0]
                    elif weight_reduction == "prod":
                        weight_imp = torch.prod(weight_imp, dim=0)
                    else:
                        raise NotImplementedError

                    weight_imp = weight_imp.item()
                    logwriter.writerow([k, weight_imp])
                    print([k, weight_imp])
                    if block_idx not in block_info.keys():
                        block_info[block_idx] = [weight_imp]
                    else:
                        block_info[block_idx].append(weight_imp)

        # Compute block-level importance
        block_info_summary = {}
        with open(result_csv_block, "w") as logfile, open(
            result_csv_block_detail, "w"
        ) as logfile_detail:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["block_name", "block_score"])
            logwriter_detail = csv.writer(logfile_detail, delimiter=",")
            logwriter_detail.writerow(["block_name", "all_weight_scores"])
            for k, v in block_info.items():
                print(k, v)
                logwriter_detail.writerow([k] + v)

                block_imp = torch.tensor(v)
                if block_reduction == "sum":
                    block_imp = block_imp.sum(dim=0)
                elif block_reduction == "mean":
                    block_imp = block_imp.mean(dim=0)
                elif block_reduction == "max":
                    block_imp = block_imp.max(dim=0)[0]
                elif block_reduction == "prod":
                    block_imp = torch.prod(block_imp, dim=0)
                else:
                    raise NotImplementedError

                block_imp = block_imp.item()
                logwriter.writerow([k, block_imp])
                block_info_summary[k] = block_imp

        for k in ["model.norm.weight", "lm_head.weight"]:
            if k in block_info_summary:
                del block_info_summary[k]
        sorted_items = sorted(block_info_summary.items(), key=lambda x: x[1])
        block_order = []
        with open(result_csv_block_sort, "w") as logfile:
            logwriter = csv.writer(logfile, delimiter=",")
            logwriter.writerow(["rank", "block_name", "block_score", "block_index"])
            for rank, (key, value) in enumerate(sorted_items, start=1):
                logwriter.writerow([rank, key, value, key.split(".")[-1]])
                print([rank, key, value, key.split(".")[-1]])
                block_order.append(int(key.split(".")[-1]))

        with open(block_order_path, "w") as logfile_order:
            logwriter_order = csv.writer(logfile_order, delimiter=",")
            logwriter_order.writerow(block_order)

        print(f"=== block order removed: {block_order_path}")
        print(block_order)
        print(f"len: {len(block_order)}")

    else:
        print(f"use the precomputed results at {block_order_path}")
