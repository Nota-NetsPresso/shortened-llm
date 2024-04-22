import os
import subprocess
from pathlib import Path

import gradio as gr
from demo import LlamaCompressionDemo


ORIG_CHECKPOINT_ID = "baffo32/decapoda-research-llama-7B-hf"
COMPRESSED_OURS_ID = "nota-ai/st-llama-1-5.5b-ppl"

S3_URL = "https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/compressed-llm/baselines/llm-pruner/llama-1-5.4b_lora_merged/pytorch_model.bin"
COMPRESSED_LLMPRUNER_ID = (
    "./src/baselines/llm-pruner/llama-1-5.4b_lora_merged/pytorch_model.bin"
)
if not os.path.exists(COMPRESSED_LLMPRUNER_ID):
    subprocess.call(f"wget {S3_URL} -O {COMPRESSED_LLMPRUNER_ID}", shell=True)

if __name__ == "__main__":
    device_orig = "cuda:0"
    device_llmprn = "cuda:0"
    device_ours = "cuda:0"

    servicer = LlamaCompressionDemo(
        ORIG_CHECKPOINT_ID,
        device_orig,
        COMPRESSED_LLMPRUNER_ID,
        device_llmprn,
        COMPRESSED_OURS_ID,
        device_ours,
    )
    example_list = servicer.get_example_list()

    with gr.Blocks(theme="nota-ai/theme") as demo:
        gr.Markdown(Path("src/docs/header.md").read_text())

        with gr.Row():
            text = gr.Textbox(
                label="Input Prompt",
                value="With recent advances in artificial intelligence (AI) and machine learning (ML) techniques,",
                max_lines=4,
                placeholder="Enter your prompt",
            )
            with gr.Accordion("Advanced Settings", open=True):
                with gr.Row():
                    max_new_tokens = gr.Slider(
                        32, 256, label="Output Tokens", value=256, step=32
                    )
                    batch_size = gr.Slider(1, 8, label="Batch Size", value=1, step=1)
                    seed = gr.Slider(0, 9999, label="Random Seed", value=1234, step=1)
                    top_k = gr.Slider(1, 1000, label="Top_k", value=50, step=1)
                    top_p = gr.Slider(0.0, 1.0, label="Top_p", value=0.95, step=0.05)
                    temperature = gr.Slider(
                        0.0, 1.0, label="Temperature", value=1.0, step=0.05
                    )
            with gr.Tab("Example Prompts"):
                examples = gr.Examples(examples=example_list, inputs=[text])

        num_params_ours = servicer.get_params(servicer.model_ours)
        with gr.Row():
            # Original model
            with gr.Column(variant="panel", scale=32):
                gr.Markdown('<h2 align="center">Original LLaMA-7B</h2>')
                gr.Markdown(f"###### {servicer.get_params(servicer.model_orig)}")
                with gr.Column():
                    gen_orig_button = gr.Button(
                        value="Generate with Original Model", variant="primary"
                    )
                    orig_model_test_time = gr.Textbox(
                        value="", label="Test Time; Throughput (Batch Size)"
                    )
                    with gr.Accordion("Output Text", open=True):
                        orig_model_output = gr.Textbox(show_label=False)
                    orig_model_output_kr = gr.Textbox(label="Korean Translation")
                    orig_model_error = gr.Markdown()

            # LLM-Pruner
            with gr.Column(variant="panel", scale=32):
                gr.Markdown('<h2 align="center">LLM-Pruner (Width✄) [NeurIPS\'23]</h2>')
                gr.Markdown(f"###### {servicer.get_params(servicer.model_ours)}")

                with gr.Column():
                    gen_llmprn_button = gr.Button(
                        value="Generate with LLM-Pruner", variant="primary"
                    )
                    llmprn_test_time = gr.Textbox(
                        value="", label="Test Time; Throughput (Batch Size)"
                    )
                    with gr.Accordion("Output Text", open=True):
                        llmprn_output = gr.Textbox(show_label=False)
                    llmprn_output_kr = gr.Textbox(label="Korean Translation")
                    llmprn_error = gr.Markdown()

            # Shortened-LLaMA (Ours)
            with gr.Column(variant="panel", scale=32):
                gr.Markdown('<h2 align="center">Ours (Depth✄) [ICLRW\'24]</h2>')
                gr.Markdown(f"###### {servicer.get_params(servicer.model_ours)}")

                with gr.Column():
                    gen_ours_button = gr.Button(
                        value="Generate with Shortened LLaMA", variant="primary"
                    )
                    ours_test_time = gr.Textbox(
                        value="", label="Test Time; Throughput (Batch Size)"
                    )
                    with gr.Accordion("Output Text", open=True):
                        ours_output = gr.Textbox(show_label=False)
                    ours_output_kr = gr.Textbox(label="Korean Translation")
                    ours_error = gr.Markdown()

        inputs = [text, max_new_tokens, batch_size, seed, top_k, top_p, temperature]

        orig_model_outputs = [
            orig_model_output,
            orig_model_test_time,
            orig_model_output_kr,
        ]
        gen_orig_button.click(
            servicer.infer_orig_model, inputs=inputs, outputs=orig_model_outputs
        )

        llmprn_outputs = [llmprn_output, llmprn_test_time, llmprn_output_kr]
        gen_llmprn_button.click(
            servicer.infer_llmpruner, inputs=inputs, outputs=llmprn_outputs
        )

        ours_outputs = [ours_output, ours_test_time, ours_output_kr]
        gen_ours_button.click(servicer.infer_ours, inputs=inputs, outputs=ours_outputs)

        gr.Markdown(Path("src/docs/footer.md").read_text())

    demo.queue(concurrency_count=1)
    demo.launch(share=True)
