import argparse
import csv
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime

import torch
from utils import count_params, get_model, set_seed


# Function to get current GPU metrics for a specific GPU index
def get_gpu_metrics(gpu_index):
    # Run the nvidia-smi command to get all the required metrics for a specific GPU
    command = f"nvidia-smi -i {gpu_index} --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)

    # Decode the result and parse the metrics
    metrics = result.stdout.decode("utf-8").strip().split(", ")
    metrics = (
        [int(metrics[0])]
        + [int(metric.strip("%")) for metric in metrics[1:3]]
        + [int(metrics[3].strip(" MiB"))]
    )

    return metrics


# Monitoring function to be run in a separate thread
def monitor_gpu(gpu_index=0):
    with open(gpu_util_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header to the CSV file
        csv_writer.writerow(
            [
                "timestamp",
                "temperature.gpu",
                "utilization.gpu [%]",
                "utilization.memory [%]",
                "memory.used [MiB]",
            ]
        )

        while monitoring_active:
            # Get current timestamp and GPU metrics
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
            gpu_metrics = get_gpu_metrics(gpu_index)

            # Write the data to the CSV file
            csv_writer.writerow([timestamp] + gpu_metrics)

            # Flush the CSV file to ensure data is written
            csvfile.flush()

            # Wait for the specified interval
            time.sleep(interval)


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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_all_runs", type=int, default=30)
    parser.add_argument("--num_warmup_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_efficiency/llama-1-7b/batch_gen_out128_bs4",
    )
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

    # Prepare input for batched generation
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        [args.input_prompt] * args.batch_size, return_tensors="pt", padding=True
    )["input_ids"].to(args.device)
    input_len = inputs[0].size(0)

    # Set log path
    os.makedirs(args.output_dir, exist_ok=True)
    gpu_util_csv = os.path.join(args.output_dir, "nvidia_smi_gpuutil.csv")
    time_stat_csv = os.path.join(args.output_dir, "latency_throughput_gpuutil.csv")
    gen_log_path = os.path.join(args.output_dir, "gen_output_gpuutil.txt")
    open(gen_log_path, "w", encoding="utf8").close()  # To remove previous record

    interval = 0.5
    monitoring_active = True
    monitoring_thread = threading.Thread(target=monitor_gpu)
    monitoring_thread.start()
    time_list = []
    throughput_list = []

    try:  # Start the GPU monitoring thread
        for i in range(args.num_all_runs):
            if "cuda" in args.device:
                starter, ender = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                starter.record()
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=inputs,
                        do_sample=True,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        max_length=(input_len + args.max_seq_len),
                        min_length=(
                            input_len + args.max_seq_len
                        ),  # forced output length (to avoid <EOS> sampling)
                        return_dict_in_generate=True,
                    )
                ender.record()
                torch.cuda.synchronize()
                batch_time = starter.elapsed_time(ender) / 1000  # in msec -> sec
            else:
                raise NotImplementedError

            if i < args.num_warmup_runs:  # no record for warmup runs
                continue
            else:
                output = tokenizer.batch_decode(generation_output.sequences)
                for b_idx, s in enumerate(output):
                    tmp_size = len(generation_output.sequences[b_idx]) - input_len
                    with open(gen_log_path, "a", encoding="utf8") as f:
                        f.write(
                            f"=== output {i} b{b_idx} | leng gen {tmp_size} + input {input_len}\n"
                        )
                        f.write(f"{s}\n")
                        print(
                            f"=== output {i} b{b_idx} | leng gen {tmp_size} + input {input_len}\n"
                        )
                        print(f"{s}\n")
                time_list.append(batch_time)
                throughput_list.append(args.batch_size * args.max_seq_len / batch_time)
    finally:  # When done, stop the monitoring
        monitoring_active = False
        monitoring_thread.join()

    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Current GPU memory occupied: {mem} MiB")
    nparams = count_params(model)
    print(f"Params: {nparams}")

    time_mean = statistics.mean(time_list)
    time_std = statistics.pstdev(time_list)

    throughput_mean = statistics.mean(throughput_list)
    throughput_std = statistics.pstdev(throughput_list)
    nbatches = len(throughput_list)

    with open(time_stat_csv, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            [
                "time_mean(sec)",
                "time_std",
                "th_mean(tokens/s)",
                "th_std",
                "mem",
                "out_len",
                "in_len",
                "nbatches",
                "batchsz",
                "nparam",
            ]
        )
        logwriter.writerow(
            [
                time_mean,
                time_std,
                throughput_mean,
                throughput_std,
                mem,
                args.max_seq_len,
                input_len,
                nbatches,
                args.batch_size,
                nparams,
            ]
        )
        logwriter.writerow(time_list)
