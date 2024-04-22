import argparse
import os
import subprocess


S3_URL = "https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/compressed-llm/tokenizer/decapoda-research-llama-7B-hf"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="tmp/tokenizer/decapoda-research-llama-7B-hf"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        raise ValueError

    for file_name in [
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.model",
    ]:
        subprocess.call(
            f"wget {S3_URL}/{file_name} -O {args.output_dir}/{file_name}", shell=True
        )
