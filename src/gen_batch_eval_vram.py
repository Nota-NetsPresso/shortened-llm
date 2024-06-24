import argparse
import csv
import gc
import os
import statistics
import threading
import tracemalloc

import psutil
import torch
from utils import count_params, get_model, set_seed


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


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
    mem_stat_csv = os.path.join(args.output_dir, "memory_tracemalloc.csv")
    time_stat_csv = os.path.join(args.output_dir, "latency_throughput_tracemalloc.csv")
    gen_log_path = os.path.join(args.output_dir, "gen_output_tracemalloc.txt")
    open(gen_log_path, "w", encoding="utf8").close()  # To remove previous record

    time_list = []
    throughput_list = []
    memory_gpu_list = []
    memory_cpu_list = []

    for i in range(args.num_all_runs):
        with TorchTracemalloc() as tracemalloc:
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
            memory_gpu_list.append(tracemalloc.peaked + b2mb(tracemalloc.begin))
            memory_cpu_list.append(tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin))

    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Current GPU memory occupied: {mem} MiB")
    nparams = count_params(model)
    print(f"Params: {nparams}")

    time_mean = statistics.mean(time_list)
    time_std = statistics.pstdev(time_list)

    throughput_mean = statistics.mean(throughput_list)
    throughput_std = statistics.pstdev(throughput_list)
    nbatches = len(throughput_list)

    mem_gpu_mean = statistics.mean(memory_gpu_list)
    mem_gpu_std = statistics.pstdev(memory_gpu_list)
    mem_cpu_mean = statistics.mean(memory_cpu_list)
    mem_cpu_std = statistics.pstdev(memory_cpu_list)

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

    with open(mem_stat_csv, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            [
                "mem_gpu_mean",
                "mem_gpu_std",
                "mem_cpu_mean",
                "mem_cpu_std",
                "out_len",
                "in_len",
                "nbatches",
                "batchsz",
                "nparam",
            ]
        )
        logwriter.writerow(
            [
                mem_gpu_mean,
                mem_gpu_std,
                mem_cpu_mean,
                mem_cpu_std,
                args.max_seq_len,
                input_len,
                nbatches,
                args.batch_size,
                nparams,
            ]
        )

    print(f"GPU peak memory: {mem_gpu_mean} MiB")
    print(f"CPU peak memory: {mem_cpu_mean} MiB")
