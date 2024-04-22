import gc
import random
import time

import googletrans
import numpy as np
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_model(model, device):
    if "cuda" in device:
        model = model.half().to(device)
    # unwind broken decapoda-research config
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    return model.eval()


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


class LlamaCompressionDemo:
    def __init__(
        self,
        orig_checkpoint_id,
        device_orig,
        compressed_llmpruner_id,
        device_llmprn,
        compressed_ours_id,
        device_ours,
    ) -> None:
        self.device_orig = device_orig
        self.device_llmprn = device_llmprn
        self.device_ours = device_ours
        self.torch_dtype = (
            torch.float16 if "cuda" in self.device_orig else torch.float32
        )
        print("\n** load models")
        start_time = time.time()
        self.model_orig = AutoModelForCausalLM.from_pretrained(
            orig_checkpoint_id, low_cpu_mem_usage=True
        )
        print(f"finish: orig model {(time.time() - start_time):.1f} sec")

        start_time = time.time()
        self.model_llmpruner = torch.load(compressed_llmpruner_id, map_location="cpu")
        self.model_llmpruner = self.model_llmpruner["model"]
        print(f"finish: llm-pruner {(time.time() - start_time):.1f} sec")

        start_time = time.time()
        self.model_ours = AutoModelForCausalLM.from_pretrained(
            compressed_ours_id, low_cpu_mem_usage=True
        )
        print(f"finish: ours {(time.time() - start_time):.1f} sec")

        self.model_orig = prepare_model(self.model_orig, self.device_orig)
        self.model_llmpruner = prepare_model(self.model_llmpruner, self.device_llmprn)
        self.model_ours = prepare_model(self.model_ours, self.device_ours)

        self.tokenizer = LlamaTokenizer.from_pretrained(orig_checkpoint_id)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        self.translator = googletrans.Translator()
        clean_memory()

    def _prepare_input(self, input_prompt, batch_size, device):
        input = [input_prompt] * int(batch_size)
        return self.tokenizer(input, return_tensors="pt", padding=True).to(device)

    def get_params(self, model):
        params_total = sum(p.numel() for p in model.parameters())
        return f"# Params: {(params_total/1e9):.1f}B"

    def generate_text(
        self,
        model,
        device,
        input_prompt,
        max_new_tokens=128,
        batch_size=1,
        top_k=50,
        top_p=0.95,
        temperature=1,
        ignore_start_token=True,
    ):
        input_tokens = self._prepare_input(input_prompt, batch_size, device)
        input_len = input_tokens["input_ids"][0].size(0)
        results_text = ""

        with torch.no_grad():
            t0 = time.perf_counter()
            generation_output = model.generate(
                input_ids=input_tokens["input_ids"],
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                max_length=(input_len + max_new_tokens),
                min_length=(input_len + max_new_tokens),
                return_dict_in_generate=True,
            )
            test_time = time.perf_counter() - t0
            test_throughput = batch_size * max_new_tokens / test_time

            output = self.tokenizer.batch_decode(generation_output.sequences)
            for b_idx, s in enumerate(output):
                print(f"** output {b_idx} ** \n{s}\n")
                if ignore_start_token:
                    results_text += f"\n{s[5:]}\n\n\n"
                else:
                    results_text += f"\n{s}\n\n\n"

        results_efficiency = f"{(test_time):.1f} sec; {(test_throughput):.1f} tokens/sec (bs {batch_size})"
        results_text_kr = self.translator.translate(results_text[:3000], dest="ko").text

        print("\n** generation done")
        print(f"{results_efficiency}")
        print(f"Input {input_len} + Output {max_new_tokens} tokens")
        print(
            f"Batch Sz {batch_size} | top_k {top_k} top_p {top_p} temperature {temperature} \n"
        )
        clean_memory()

        return results_text, results_efficiency, results_text_kr

    def infer_orig_model(
        self, input_prompt, max_new_tokens, batch_size, seed, top_k, top_p, temperature
    ):
        set_seed(seed)
        print(f"=== ORIG model | seed {seed}")
        print(input_prompt, max_new_tokens, batch_size, seed, top_k, top_p, temperature)
        output_text, time_throughput_txt, results_text_kr = self.generate_text(
            self.model_orig,
            self.device_orig,
            input_prompt,
            max_new_tokens,
            batch_size,
            top_k,
            top_p,
            temperature,
        )
        return output_text, time_throughput_txt, results_text_kr

    def infer_llmpruner(
        self, input_prompt, max_new_tokens, batch_size, seed, top_k, top_p, temperature
    ):
        set_seed(seed)
        print(f"=== COMPRESSED LLM-Pruner | seed {seed}")
        output_text, time_throughput_txt, results_text_kr = self.generate_text(
            self.model_llmpruner,
            self.device_llmprn,
            input_prompt,
            max_new_tokens,
            batch_size,
            top_k,
            top_p,
            temperature,
        )
        return output_text, time_throughput_txt, results_text_kr

    def infer_ours(
        self, input_prompt, max_new_tokens, batch_size, seed, top_k, top_p, temperature
    ):
        set_seed(seed)
        print(f"=== COMPRESSED Ours  | seed {seed}")
        output_text, time_throughput_txt, results_text_kr = self.generate_text(
            self.model_ours,
            self.device_ours,
            input_prompt,
            max_new_tokens,
            batch_size,
            top_k,
            top_p,
            temperature,
        )
        return output_text, time_throughput_txt, results_text_kr

    def get_example_list(self):
        return [
            "AI can create a logo in seconds",
            "What’s great about the holiday season,",
            "Neural network pruning is defined as",
            "The Leaning Tower of Pisa is known for",
        ]
