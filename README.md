# Shortened LLM by Nota AI
Official codebase for [**Shortened LLaMA: A Simple Depth Pruning for Large Language Models**](https://openreview.net/forum?id=18VGxuOdpu) [[ArXiv](https://arxiv.org/abs/2402.02834)] [[ICLR 2024 Workshop on ME-FoMo](https://sites.google.com/view/me-fomo2024)].

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

## Examples
The scripts perform (1) block pruning ➔ (2) LoRA-based retraining ➔ (3) zero-shot evaluation.
- Pruning criterion: PPL (top); Taylor+ (bottom).
- 20% pruning of [LLaMA-1-7b](https://huggingface.co/baffo32/decapoda-research-llama-7B-hf)
  ```bash
  bash script/prune_llama-7b_crit-ppl.sh
  bash script/prune_llama-7b_crit-taylor.sh
  ```
- 20% pruning of [Vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3)
  ```bash
  bash script/prune_vicuna-7b_crit-ppl.sh
  bash script/prune_vicuna-7b_crit-taylor.sh
  ```
- 21% pruning of [Vicuna-13b-v1.3](https://huggingface.co/lmsys/vicuna-13b-v1.3) 
  ```bash
  bash script/prune_vicuna-13b_crit-ppl.sh
  bash script/prune_vicuna-13b_crit-taylor.sh
  ```

## Model Description
After identifying unimportant Transformer blocks, we perform one-shot pruning and light LoRA-based retraining.
    <details>
    <summary>
    Click to see a method figure:
    </summary>
    <img alt="method" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/compressed-llm/st-llama_method.png" width="100%">
    </details>

#### Model Links
- Available at 🤗Hugging Face Models:

  | Source<br>Model | Pruning<br>Ratio | Pruning<br>Criterion | HF Models<br>Link |
  |:---:|:---:|:---:|:---:|
  | LLaMA-1-7B | 20% | PPL | [nota-ai/st-llama-1-5.5b-ppl](https://huggingface.co/nota-ai/st-llama-1-5.5b-ppl) |
  | LLaMA-1-7B | 20% | Taylor+ | [nota-ai/st-llama-1-5.5b-taylor](https://huggingface.co/nota-ai/st-llama-1-5.5b-taylor) |
  | Vicuna-v1.3-7B | 20% | PPL | [nota-ai/st-vicuna-v1.3-5.5b-ppl](https://huggingface.co/nota-ai/st-vicuna-v1.3-5.5b-ppl) |
  | Vicuna-v1.3-7B | 20% | Taylor+ | [nota-ai/st-vicuna-v1.3-5.5b-taylor](https://huggingface.co/nota-ai/st-vicuna-v1.3-5.5b-taylor) |
  | Vicuna-v1.3-13B | 21% | PPL | [nota-ai/st-vicuna-v1.3-10.5b-ppl](https://huggingface.co/nota-ai/st-vicuna-v1.3-10.5b-ppl) |
  | Vicuna-v1.3-13B | 21% | Taylor+ | [nota-ai/st-vicuna-v1.3-10.5b-taylor](https://huggingface.co/nota-ai/st-vicuna-v1.3-10.5b-taylor) |

#### Zero-shot Evaluation
- To measure (1) PPL on WikiText2 & PTB, and (2) accuracy on seven commonsense reasoning tasks, use: (EleutherAI/lm-evaluation-harness version [3326c54](https://github.com/EleutherAI/lm-evaluation-harness/tree/3326c547a733d598b4377e54be96e194861b964c))
    ```bash
    bash script/evaluate.sh
    ```

    <details>
    <summary>
    Click to see the zero-shot results:
    </summary>
  <img alt="results" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/compressed-llm/st-llama_zero-shot_scores.png" width="100%">
    </details>

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
- Meta AI's [LLaMA](https://github.com/facebookresearch/llama) and  LMSYS Org's [Vicuna](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md). Thanks for the open-source LLMs!

## Citation
```bibtex
@article{kim2024shortened,
  title={Shortened LLaMA: A Simple Depth Pruning for Large Language Models},
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
