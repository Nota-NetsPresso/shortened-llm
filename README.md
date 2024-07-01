# Shortened LLM by Nota AI
Official codebase for [**Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods**](https://arxiv.org/abs/2402.02834) [[ArXiv](https://arxiv.org/abs/2402.02834)] [[ICLR 2024 Workshop on ME-FoMo](https://sites.google.com/view/me-fomo2024)][[Blog Post](https://www.nota.ai/community/shortened-llm-a-simple-depth-pruning-for-large-language-models)].

* We perform one-shot pruning by removing unimportant Transformer blocks in LLMs. Compared to recent baselines, our **depth pruning** achieves faster inference while yielding comparable or superior performance.
* In retraining pruned models for quality recovery, **continued pretraining (CPT)** on a large corpus markedly outperforms LoRA-based tuning, particularly at severe pruning ratios.


<p align="center">
<img alt="teaser" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/compressed-llm/teaser.png" width="60%">
</p>


## Installation
  ```bash
  conda create -n shortened-llm python=3.9
  conda activate shortened-llm
  git clone https://github.com/Nota-NetsPresso/shortened-llm.git
  cd shortened-llm
  pip install -r requirement.txt
  ```

<details>
<summary>
Note on package versions:
</summary>


- Part of the below repositories is included for evaluation:
  - `src/LLMPruner`: horseee/LLM-Pruner version [213ffa4](https://github.com/horseee/LLM-Pruner/tree/213ffa4d02f92f16d29219a97fd01a8622db1550)
  - `src/lm_eval`: EleutherAI/lm-evaluation-harness version [3326c54](https://github.com/EleutherAI/lm-evaluation-harness/tree/3326c547a733d598b4377e54be96e194861b964c)
- Torch version used in our experiments: `2.0.1` for RTX3090 & A100; `2.1.1` for H100. 

</details>

<details>
<summary>
(optional) GPTQ Support:
</summary>


- Post-training quantization can be further applied to our pruned model. 
- We applied GPTQ on the pruned & re-trained models.
  - repo: [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ/tree/v0.7.1) version `0.7.1`
- To install the required packages, we recommend installation from source as follows:
  ```bash
  git clone https://github.com/AutoGPTQ/AutoGPTQ.git
  cd AutoGPTQ
  git checkout v0.7.1
  pip install -vvv -e .
  ```

</details>



## Models from Aggressive Pruning & CPT Retraining (arXiv-v2):
  | Source<br>Model | Pruning<br>Ratio | Pruning<br>Criterion | 🤗Hugging Face<br>Link |
  |:---:|:---:|:---:|:---:|
  | Vicuna-v1.3-7B | 20% | PPL | [nota-ai/cpt_st-vicuna-v1.3-5.5b-ppl](https://huggingface.co/nota-ai/cpt_st-vicuna-v1.3-5.5b-ppl) |
  | Vicuna-v1.3-7B | 45% | PPL | [nota-ai/cpt_st-vicuna-v1.3-3.7b-ppl](https://huggingface.co/nota-ai/cpt_st-vicuna-v1.3-3.7b-ppl) |
  | Vicuna-v1.3-7B | 60% | PPL | [nota-ai/cpt_st-vicuna-v1.3-2.7b-ppl](https://huggingface.co/nota-ai/cpt_st-vicuna-v1.3-2.7b-ppl) |
  | Vicuna-v1.3-7B | 80% | PPL | [nota-ai/cpt_st-vicuna-v1.3-1.5b-ppl](https://huggingface.co/nota-ai/cpt_st-vicuna-v1.3-1.5b-ppl) |

<details>
<summary>
Click to see the results:
</summary>

- EleutherAI/lm-evaluation-harness version [3326c54](https://github.com/EleutherAI/lm-evaluation-harness/tree/3326c547a733d598b4377e54be96e194861b964c)

<img alt="results" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/compressed-llm/st_llm-cpt_results.png" width="100%">

</details>


## Models from Moderate Pruning & LoRA Retraining (arXiv-v1):
  | Source<br>Model | Pruning<br>Ratio | Pruning<br>Criterion | 🤗Hugging Face<br>Link |
  |:---:|:---:|:---:|:---:|
  | LLaMA-1-7B | 20% | PPL | [nota-ai/st-llama-1-5.5b-ppl](https://huggingface.co/nota-ai/st-llama-1-5.5b-ppl) |
  | LLaMA-1-7B | 20% | Taylor+ | [nota-ai/st-llama-1-5.5b-taylor](https://huggingface.co/nota-ai/st-llama-1-5.5b-taylor) |
  | Vicuna-v1.3-7B | 20% | PPL | [nota-ai/st-vicuna-v1.3-5.5b-ppl](https://huggingface.co/nota-ai/st-vicuna-v1.3-5.5b-ppl) |
  | Vicuna-v1.3-7B | 20% | Taylor+ | [nota-ai/st-vicuna-v1.3-5.5b-taylor](https://huggingface.co/nota-ai/st-vicuna-v1.3-5.5b-taylor) |
  | Vicuna-v1.3-13B | 21% | PPL | [nota-ai/st-vicuna-v1.3-10.5b-ppl](https://huggingface.co/nota-ai/st-vicuna-v1.3-10.5b-ppl) |
  | Vicuna-v1.3-13B | 21% | Taylor+ | [nota-ai/st-vicuna-v1.3-10.5b-taylor](https://huggingface.co/nota-ai/st-vicuna-v1.3-10.5b-taylor) |

<details>

<summary>
Click to see the results:
</summary>

- EleutherAI/lm-evaluation-harness version [3326c54](https://github.com/EleutherAI/lm-evaluation-harness/tree/3326c547a733d598b4377e54be96e194861b964c)

<img alt="results" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/compressed-llm/st-llama_zero-shot_scores.png" width="100%">

</details>


## Examples
The scripts perform (1) block pruning ➔ (2) LoRA-based retraining ➔ (3) zero-shot evaluation.
- Pruning criterion: PPL (top); Taylor+ (bottom).
- [LLaMA-1-7b](https://huggingface.co/baffo32/decapoda-research-llama-7B-hf) (based on `LlamaForCausalLM`)
  ```bash
  bash script/prune_llama-7b_crit-ppl.sh
  bash script/prune_llama-7b_crit-taylor.sh
  ```
- [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) (based on `LlamaForCausalLM`) 
  ```bash
  bash script/prune_llama2-7b_crit-ppl.sh
  bash script/prune_llama2-7b_crit-taylor.sh
  ```
- [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (based on `LlamaForCausalLM`) 
  ```bash
  bash script/prune_llama3-8b_crit-ppl.sh
  bash script/prune_llama3-8b_crit-taylor.sh
  ```
- [Vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3) (based on `LlamaForCausalLM`)
  ```bash
  bash script/prune_vicuna-7b_crit-ppl.sh
  bash script/prune_vicuna-7b_crit-taylor.sh
  ```
- [Vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3) (based on `LlamaForCausalLM`) 
  ```bash
  bash script/prune_vicuna-13b_crit-ppl.sh
  bash script/prune_vicuna-13b_crit-taylor.sh
  ```
- [CatPPT-base](https://huggingface.co/rishiraj/CatPPT-base) (based on `MistralForCausalLM`)
  ```bash
  bash script/prune_CatPPT_crit-ppl.sh
  bash script/prune_CatPPT_crit-taylor.sh
  ```
- [Gemma-2b](https://huggingface.co/google/gemma-2b) (based on `GemmaForCausalLM`)
  ```bash
  bash script/prune_gemma-2b_crit-ppl_yesBOS.sh
  bash script/prune_gemma-2b_crit-taylor_yesBOS.sh
  ```
- [Gemma-7b](https://huggingface.co/google/gemma-7b) (based on `GemmaForCausalLM`) 
  ```bash
  bash script/prune_gemma-7b_crit-ppl_yesBOS.sh
  bash script/prune_gemma-7b_crit-taylor_yesBOS.sh
  ```


## Other Scripts
- To test other pruning ratios, use:
  ```bash
  bash script/prune.sh
  ```

- To obtain baselines using the magnitude pruning criterion, use:
  ```bash
  bash script/prune_llama-7b_crit-magnitude.sh
  bash script/prune_vicuna-7b_crit-magnitude.sh
  bash script/prune_vicuna-13b_crit-magnitude.sh
  ```

- To measure (1) PPL on WikiText2 & PTB, and (2) accuracy on seven commonsense reasoning tasks, use: (EleutherAI/lm-evaluation-harness version [3326c54](https://github.com/EleutherAI/lm-evaluation-harness/tree/3326c547a733d598b4377e54be96e194861b964c))
  ```bash
  bash script/evaluate.sh
  ```

- (Optional) Any post-training quantization method can be applied to our pruned models. The example script quantizes our pruned models using GPTQ and measures their performance with `script/evaluate.sh`:
  ```bash
  bash script/quantize_gptq_vicuna-7b.sh
  ```

- To measure latency & throughput, use:
  ```bash
  bash script/measure_time.sh
  ```

- To measure VRAM requirements, use:
  ```bash
  bash script/measure_vram.sh
  ```

- To measure GPU compute utilization, use:
  ```bash
  bash script/measure_gpuutil.sh
  ```

## Gradio Demo: Width✄ vs. Depth✄
The demo compares the use of [LLM-Pruner](https://arxiv.org/abs/2305.11627) (Ma et al., 2023; width pruning) and [Shortened LLaMA](https://arxiv.org/abs/2402.02834) (Ours; depth pruning) for the LLaMA-1-7B model:
  ```bash
  pip install transformers==4.33.1 # to run LLM-Pruner's model
  python src/app.py
  ```
<details>
<summary>
Click to see a demo screenshot (on an A100 80GB GPU):
</summary>
<img alt="demo" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/compressed-llm/st-llama_demo_screenshot.png" width="100%">
</details>


## License
- All rights related to this repository and the compressed models are reserved by Nota Inc.
- The intended use is strictly limited to research and non-commercial projects.

## Acknowledgments
- [Microsoft for Startups Founders Hub](https://www.microsoft.com/en-us/startups) and [Gwangju AICA](http://www.aica-gj.kr/main.php) for generously providing GPU resources.
- [LLM-Pruner](https://github.com/horseee/LLM-Pruner), which utilizes [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), [PEFT](https://github.com/huggingface/peft), and [Alpaca-LoRA](https://github.com/tloen/alpaca-lora). Thanks for the pioneering work on structured pruning of LLMs! 
- [LLaMA](https://github.com/facebookresearch/llama), [Vicuna](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md), and [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B). Thanks for the open-source LLMs and data!

## Citation
```bibtex
@article{kim2024shortened,
  title={Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods},
  author={Kim, Bo-Kyeong and Kim, Geonmin and Kim, Tae-Ho and Castells, Thibault and Choi, Shinkook and Shin, Junho and Song, Hyoung-Kyu},
  journal={arXiv preprint arXiv:2402.02834},      
  year={2024},
  url={https://arxiv.org/abs/2402.02834}
}
```
```bibtex
@article{kim2024mefomo,
  title={Shortened LLaMA: A Simple Depth Pruning for Large Language Models},
  author={Kim, Bo-Kyeong and Kim, Geonmin and Kim, Tae-Ho and Castells, Thibault and Choi, Shinkook and Shin, Junho and Song, Hyoung-Kyu},
  journal={ICLR Workshop on Mathematical and Empirical Understanding of Foundation Models (ME-FoMo)},
  year={2024},
  url={https://openreview.net/forum?id=18VGxuOdpu}
}
```
